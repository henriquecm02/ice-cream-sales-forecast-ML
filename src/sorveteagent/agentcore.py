"""Núcleo do agente: cliente Gemini, laço de ferramentas, conferência de números e aprovações.

O padrão é o mesmo dos outros projetos do portfólio:
- o modelo interpreta a pergunta e escolhe ferramentas (saída em JSON Schema);
- as ferramentas são código determinístico (pandas/sklearn) e o modelo nunca calcula;
- o código confere se todo número citado na resposta saiu de uma ferramenta;
- ferramentas sensíveis não rodam sozinhas: viram ações pendentes de aprovação humana.

Sem GEMINI_API_KEY o agente roda em modo offline, com roteamento por palavras-chave
e resposta montada a partir dos resumos das ferramentas.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


# --------------------------------------------------------------------------- LLM
class QuotaExceeded(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def json(self, system: str, user: str, schema: dict) -> dict: ...


class GeminiLLM:
    """Chamada ao Gemini com saída estruturada (JSON Schema) e temperatura 0."""

    def __init__(self, api_key: str, model: str | None = None, retries: int = 3):
        from google import genai  # import tardio: os testes não precisam do SDK

        self.client = genai.Client(api_key=api_key)
        self.name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.retries = retries

    def json(self, system: str, user: str, schema: dict) -> dict:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0,
        )
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self.client.models.generate_content(model=self.name, contents=user, config=config)
                return json.loads(resp.text)
            except Exception as exc:
                last = exc
                if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                    raise QuotaExceeded(str(exc)) from exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Falha ao chamar o Gemini: {last}")


class ScriptedLLM:
    """LLM de teste: devolve respostas pré-definidas, em ordem ou via função."""

    name = "scripted"

    def __init__(self, responses: list[dict] | Callable[[str, str, dict], dict]):
        self._responses = responses
        self.calls: list[dict] = []

    def json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append({"system": system, "user": user})
        if callable(self._responses):
            return self._responses(system, user, schema)
        return self._responses.pop(0)


def load_env() -> None:
    """Lê GEMINI_API_KEY de ./.env ou ~/.config/portfolio/.env, sem sobrescrever o ambiente."""
    for path in (Path(".env"), Path.home() / ".config" / "portfolio" / ".env"):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_llm() -> LLM | None:
    load_env()
    key = os.getenv("GEMINI_API_KEY")
    return GeminiLLM(key) if key else None


# --------------------------------------------------------------------- Ferramentas
@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]  # nome -> descrição (tipo e padrão)
    fn: Callable[..., dict]
    sensitive: bool = False  # True: exige aprovação humana antes de executar

    def run(self, **kwargs: Any) -> dict:
        return self.fn(**kwargs)


@dataclass
class PendingAction:
    tool: str
    args: dict
    reason: str


@dataclass
class AgentResult:
    question: str
    mode: str  # "gemini" ou "offline"
    calls: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    answer: str = ""
    numbers_cited: list[float] = field(default_factory=list)
    ungrounded: list[float] = field(default_factory=list)
    pending: list[PendingAction] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.errors:
            return "erro"
        return "revisar" if self.ungrounded else "ok"


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"tool": {"type": "string"}, "args_json": {"type": "string"}},
                "required": ["tool", "args_json"],
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["calls"],
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "numbers_cited": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["answer", "numbers_cited"],
}

PLAN_SYSTEM = """Você é um agente de análise de dados. Escolha as ferramentas necessárias para responder
à pergunta do usuário e preencha os argumentos em JSON (args_json). Use apenas ferramentas do catálogo.
Nunca invente números: eles virão das ferramentas. Ferramentas marcadas como SENSÍVEL só podem ser
propostas; uma pessoa vai aprovar antes de rodar. Se a pergunta não puder ser respondida com o
catálogo, devolva calls vazio.
Catálogo:
{catalog}"""

ANSWER_SYSTEM = """Você responde em português do Brasil, de forma curta e direta, usando SOMENTE os
números que aparecem nos resultados das ferramentas. Liste em numbers_cited todo número que aparecer
na resposta. Se uma ação ficou pendente de aprovação, diga isso. Não invente dados."""


# ------------------------------------------------------------- Conferência de números
_NUM = re.compile(r"-?\d+(?:[.,]\d+)*(?:[.,]\d+)?")


def parse_number(token: str) -> float | None:
    """Converte '1.234,5', '1,234.5', '12,5', '12.5' ou '1234' em float."""
    t = token.strip()
    if not t:
        return None
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif "," in t:
        parts = t.split(",")
        t = t.replace(",", "") if len(parts[-1]) == 3 and len(parts) > 1 and len(parts[0]) <= 3 else t.replace(",", ".")
    elif t.count(".") > 1 or (t.count(".") == 1 and len(t.split(".")[-1]) == 3 and len(t.split(".")[0]) <= 3 and len(t) > 4):
        t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None


def numbers_in_text(text: str) -> list[float]:
    out = []
    for tok in _NUM.findall(text):
        v = parse_number(tok)
        if v is not None:
            out.append(v)
    return out


def _collect_numbers(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(numbers_in_text(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)


def ungrounded_numbers(cited: list[float], evidence: Any, question: str = "", tol: float = 0.005) -> list[float]:
    """Números citados que não batem com nenhum número da evidência.

    Tolerância: 0,5% do valor ou 0,051 em termos absolutos (arredondamento para uma casa).
    A única conversão aceita é fração -> porcentagem (0,35 citado como 35%)."""
    pool: list[float] = []
    _collect_numbers(evidence, pool)
    pool.extend(numbers_in_text(question))
    bad = []
    for c in cited:
        if abs(c) <= 1 and float(c).is_integer():
            continue  # 0 e 1 aparecem em qualquer texto
        ok = False
        for p in pool:
            for cand in (p, p * 100) if abs(p) <= 1 else (p,):
                if abs(c - cand) <= max(tol * abs(cand), 0.051):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            bad.append(c)
    return bad


# ------------------------------------------------------------------------ Agente
class Agent:
    def __init__(
        self,
        tools: list[Tool],
        llm: LLM | None = None,
        offline_router: Callable[[str], list[dict]] | None = None,
        max_calls: int = 4,
    ):
        self.tools = {t.name: t for t in tools}
        self.llm = llm
        self.offline_router = offline_router
        self.max_calls = max_calls

    def catalog(self) -> str:
        lines = []
        for t in self.tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in t.params.items()) or "sem argumentos"
            flag = " [SENSÍVEL: exige aprovação]" if t.sensitive else ""
            lines.append(f"- {t.name}({params}): {t.description}{flag}")
        return "\n".join(lines)

    def _plan(self, question: str) -> list[dict]:
        if self.llm is None:
            return self.offline_router(question) if self.offline_router else []
        plan = self.llm.json(PLAN_SYSTEM.format(catalog=self.catalog()), question, PLAN_SCHEMA)
        calls = []
        for c in plan.get("calls", [])[: self.max_calls]:
            try:
                args = json.loads(c.get("args_json") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"tool": c.get("tool", ""), "args": args if isinstance(args, dict) else {}})
        return calls

    def run(self, question: str) -> AgentResult:
        res = AgentResult(question=question, mode="gemini" if self.llm else "offline")
        try:
            res.calls = self._plan(question)
        except QuotaExceeded as exc:
            res.errors.append(f"Cota do Gemini esgotada: {exc}")
            return res
        for call in res.calls:
            tool = self.tools.get(call["tool"])
            if tool is None:
                res.errors.append(f"Ferramenta desconhecida: {call['tool']}")
                continue
            if tool.sensitive:
                res.pending.append(PendingAction(tool.name, call["args"], "ação sensível: aguarda aprovação"))
                continue
            try:
                out = tool.run(**call["args"])
            except TypeError as exc:
                res.errors.append(f"Argumentos inválidos para {tool.name}: {exc}")
                continue
            except ValueError as exc:
                res.errors.append(f"{tool.name}: {exc}")
                continue
            res.results.append({"tool": tool.name, "args": call["args"], "output": out})

        if not res.calls:
            res.answer = "Não encontrei uma ferramenta que responda a essa pergunta. Tente reformular."
            return res
        if not res.results and not res.pending:
            res.answer = "Não consegui rodar as ferramentas: " + "; ".join(res.errors)
            return res

        if self.llm is None:
            parts = [r["output"].get("resumo", "") for r in res.results]
            parts += [f"Ação pendente de aprovação: {p.tool}." for p in res.pending]
            res.answer = "\n\n".join(p for p in parts if p)
            res.numbers_cited = numbers_in_text(res.answer)
        else:
            evidence = json.dumps(
                {"resultados": res.results, "pendentes": [p.__dict__ for p in res.pending]},
                ensure_ascii=False, default=str,
            )[:30000]
            ans = self.llm.json(ANSWER_SYSTEM, f"Pergunta: {question}\n\nResultados:\n{evidence}", ANSWER_SCHEMA)
            res.answer = ans.get("answer", "")
            res.numbers_cited = [float(x) for x in ans.get("numbers_cited", [])] + numbers_in_text(res.answer)

        res.ungrounded = ungrounded_numbers(res.numbers_cited, res.results, question)
        return res

    def approve(self, action: PendingAction) -> dict:
        """Executa uma ação sensível depois da aprovação humana."""
        return self.tools[action.tool].run(**action.args)
