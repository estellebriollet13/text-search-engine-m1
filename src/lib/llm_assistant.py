import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path


LLM_MAX_REFORMULATIONS = 3
LLM_TOP_RESULTS = 2
LLM_ABSTRACT_MAX_CHARS = 700
DEFAULT_LLM_TIMEOUT = 60
DEFAULT_LLM_MAX_TOKENS = 1024


def run_llm_search_assistant(
    patents,
    user_query,
    search_mode,
    llm_config,
    search_callback,
    clean_value,
):
    attempted_queries = []
    current_query = user_query.strip()

    for attempt_number in range(1, LLM_MAX_REFORMULATIONS + 2):
        attempted_queries.append(current_query)
        search_results = search_callback(current_query)
        top_results = search_results[:LLM_TOP_RESULTS]
        patent_context = build_llm_patent_context(top_results, patents, clean_value)

        assessment = assess_llm_search_results(
            llm_config=llm_config,
            original_query=user_query,
            current_query=current_query,
            patent_context=patent_context,
            attempted_queries=attempted_queries,
            can_reformulate=attempt_number <= LLM_MAX_REFORMULATIONS,
            clean_value=clean_value,
        )

        if assessment.get("relevant"):
            final_answer = build_llm_final_answer(
                llm_config=llm_config,
                original_query=user_query,
                final_query=current_query,
                patent_context=patent_context,
            )
            return {
                "status": "success",
                "answer": final_answer,
                "attempts": attempted_queries,
                "results": top_results,
                "search_mode": search_mode,
                "final_query": current_query,
                "assessment": assessment,
            }

        reformulated_query = clean_value(assessment.get("reformulated_query"))
        if attempt_number > LLM_MAX_REFORMULATIONS or not reformulated_query:
            return {
                "status": "failed",
                "answer": clean_value(assessment.get("explanation")) or (
                    "Aucun ensemble de brevets suffisamment pertinent n'a été trouvé "
                    "après les reformulations autorisées."
                ),
                "attempts": attempted_queries,
                "results": top_results,
                "search_mode": search_mode,
                "final_query": current_query,
                "assessment": assessment,
            }

        current_query = make_reformulation_unique(
            reformulated_query,
            attempted_queries,
            fallback_seed=user_query,
            attempt_number=attempt_number,
            clean_value=clean_value,
        )

    return {
        "status": "failed",
        "answer": "La recherche s'est arrêtée sans résultat suffisamment pertinent.",
        "attempts": attempted_queries,
        "results": [],
        "search_mode": search_mode,
        "final_query": current_query,
        "assessment": {},
    }


def build_llm_patent_context(results, patents, clean_value):
    context = []
    for rank, (doc_id, score) in enumerate(results, start=1):
        patent = patents.get(doc_id, {})
        context.append(
            {
                "rank": rank,
                "score": round(float(score), 6),
                "patent_id": patent.get("patent_id", ""),
                "title": patent.get("title", ""),
                "abstract": truncate_text(patent.get("abstract", ""), LLM_ABSTRACT_MAX_CHARS, clean_value),
            }
        )
    return context


def assess_llm_search_results(
    llm_config,
    original_query,
    current_query,
    patent_context,
    attempted_queries,
    can_reformulate,
    clean_value,
):
    system_prompt = (
        "You are a linguistic relevance checker for a patent search engine. "
        "Your goal is to decide whether the patent titles and abstracts match the user's query, "
        "even when the keywords differ or are synonyms. Be concise. Do not add chatter. "
        "Give one global relevance score from 0 to 10 and explain the conceptual link in one sentence. "
        "If the score is below 6/10 and reformulation is allowed, propose exactly one short reformulated "
        "search query in English, different from the queries already tested. "
        "The patent corpus is in English, so every reformulated search query must be in English patent-search terms, "
        "even if the user's original query is in another language. "
        "Return pure valid JSON only. Do not wrap the JSON in markdown or ```json code fences. "
        "Do not write any free-form text outside the JSON object."
    )
    user_payload = {
        "user_query": original_query,
        "current_search_query": current_query,
        "already_tested_queries": attempted_queries,
        "reformulation_allowed": can_reformulate,
        "patents_to_analyze": [
            {
                "rank": patent["rank"],
                "title": patent["title"],
                "abstract": patent["abstract"],
            }
            for patent in patent_context
        ],
        "expected_json_format": {
            "relevance_score": "number from 0 to 10",
            "relevant": "boolean, true if relevance_score >= 6",
            "explanation": "one sentence explaining the conceptual link",
            "answer": "",
            "reformulated_query": "English search query, empty if relevant=true or no useful reformulation exists",
        },
    }
    try:
        response = call_chat_llm(
            llm_config,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=0.1,
        )
        return parse_llm_json(response, clean_value)
    except Exception as e:
        import streamlit as st

        print(f"[ERREUR CRITIQUE ASSESS]: {e}")
        st.error(f"Détail du crash : {e}")
        return {
            "relevance_score": 0.0,
            "relevant": False,
            "explanation": str(e),
            "answer": "",
            "reformulated_query": "",
        }


def build_llm_final_answer(llm_config, original_query, final_query, patent_context):
    system_prompt = (
        "Tu es un assistant de recherche brevet. A partir des brevets fournis, "
        "rédige une réponse naturelle en français. Explique brièvement pourquoi "
        "les brevets retenus semblent pertinents. Ne prétends pas avoir lu autre chose "
        "que les titres et abstracts fournis."
    )
    user_payload = {
        "original_user_query": original_query,
        "search_query_used": final_query,
        "patents": patent_context,
        "answer_requirements": [
            "Présenter les brevets les plus pertinents.",
            "Inclure le Patent ID quand il est disponible.",
            "Expliquer en une phrase pourquoi chaque brevet est lié au besoin.",
            "Signaler clairement si la pertinence reste incertaine.",
        ],
    }
    return call_chat_llm(
        llm_config,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0.2,
    )


def load_llm_config(env_path):
    env_path = Path(env_path)
    values = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")

    merged = {**values, **os.environ}
    return {
        "url": (
            merged.get("LLM_URL")
            or merged.get("OLLAMA_URL")
            or merged.get("OPENAI_BASE_URL")
            or merged.get("URL")
            or ""
        ).strip(),
        "model": (
            merged.get("LLM_MODEL")
            or merged.get("OLLAMA_MODEL")
            or merged.get("OPENAI_MODEL")
            or merged.get("MODEL")
            or ""
        ).strip(),
        "api_key": (
            merged.get("LLM_API_KEY")
            or merged.get("OPENAI_API_KEY")
            or merged.get("API_KEY")
            or ""
        ).strip(),
        "api_type": (merged.get("LLM_API_TYPE") or merged.get("API_TYPE") or "auto").strip().lower(),
        "timeout": parse_positive_int(merged.get("LLM_TIMEOUT"), DEFAULT_LLM_TIMEOUT),
        "max_tokens": parse_positive_int(merged.get("LLM_MAX_TOKENS"), DEFAULT_LLM_MAX_TOKENS),
    }


def parse_positive_int(value, default):
    try:
        parsed_value = int(str(value or "").strip())
    except (TypeError, ValueError):
        return default
    return parsed_value if parsed_value > 0 else default


def call_chat_llm(llm_config, messages, temperature=0.2):
    api_type = infer_llm_api_type(llm_config["url"], llm_config["api_type"])
    if api_type == "anthropic":
        return call_messages_llm(llm_config, messages, temperature=temperature)

    if api_type == "ollama":
        endpoint = build_ollama_chat_url(llm_config["url"])
        payload = {
            "model": llm_config["model"],
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": llm_config.get("max_tokens", DEFAULT_LLM_MAX_TOKENS),
            },
        }
    else:
        endpoint = build_openai_chat_url(llm_config["url"])
        payload = {
            "model": llm_config["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": llm_config.get("max_tokens", DEFAULT_LLM_MAX_TOKENS),
        }

    headers = {"Content-Type": "application/json"}
    if llm_config["api_key"]:
        headers["Authorization"] = f"Bearer {llm_config['api_key']}"
        headers["x-api-key"] = llm_config["api_key"]

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=llm_config.get("timeout", DEFAULT_LLM_TIMEOUT),
        ) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Erreur HTTP {error.code} depuis {endpoint}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Impossible de joindre l'API LLM à {endpoint}: {error.reason}") from error

    if api_type == "ollama":
        return (response_payload.get("message", {}).get("content") or "").strip()

    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError("Réponse LLM invalide : aucun choix retourné.")
    return (choices[0].get("message", {}).get("content") or "").strip()


def call_messages_llm(llm_config, messages, temperature=0.2):
    endpoint = build_messages_url(llm_config["url"])
    payload = {
        "model": llm_config["model"],
        "max_tokens": llm_config.get("max_tokens", DEFAULT_LLM_MAX_TOKENS),
        "temperature": temperature,
        "stream": False,
        "messages": convert_chat_messages_to_messages_api(messages),
    }
    headers = {"Content-Type": "application/json"}
    if llm_config["api_key"]:
        headers["x-api-key"] = llm_config["api_key"]
        headers["Authorization"] = f"Bearer {llm_config['api_key']}"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=llm_config.get("timeout", DEFAULT_LLM_TIMEOUT),
        ) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Erreur HTTP {error.code} depuis {endpoint}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Impossible de joindre l'API LLM à {endpoint}: {error.reason}") from error

    return extract_messages_response_text(response_payload)


def convert_chat_messages_to_messages_api(messages):
    system_parts = []
    conversation_parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            conversation_parts.append(f"[{role.upper()}]\n{content}")

    content_parts = []
    if system_parts:
        content_parts.append("[SYSTEM]\n" + "\n\n".join(system_parts))
    content_parts.extend(conversation_parts)
    return [{"role": "user", "content": "\n\n".join(content_parts)}]


def extract_messages_response_text(response_payload):
    content = response_payload.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(part for part in text_parts if part).strip()

    message = response_payload.get("message", {})
    if isinstance(message, dict):
        return (message.get("content") or "").strip()
    return ""


def infer_llm_api_type(url, configured_type):
    if configured_type in {"anthropic", "messages", "jan_messages"}:
        return "anthropic"
    if configured_type in {"ollama", "openai"}:
        return configured_type
    normalized_url = url.lower()
    if "/api" in normalized_url and "/v1" not in normalized_url:
        return "ollama"
    return "openai"


def build_ollama_chat_url(url):
    base_url = url.rstrip("/")
    if base_url.endswith("/api/chat"):
        return base_url
    if base_url.endswith("/api"):
        return f"{base_url}/chat"
    return f"{base_url}/api/chat"


def build_openai_chat_url(url):
    base_url = url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def build_messages_url(url):
    base_url = url.rstrip("/")
    if base_url.endswith("/messages"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/messages"
    return f"{base_url}/v1/messages"


def parse_llm_json(raw_response, clean_value):
    cleaned_response = strip_markdown_json_fence(raw_response)
    try:
        parsed = json.loads(cleaned_response)
    except json.JSONDecodeError:
        try:
            match = re.search(r"\{.*\}", cleaned_response, flags=re.DOTALL)
            if not match:
                return build_llm_json_fallback(cleaned_response)
            parsed = json.loads(match.group(0))
        except (TypeError, json.JSONDecodeError, re.error):
            return build_llm_json_fallback(cleaned_response)

    relevance_score = parsed.get("relevance_score", parsed.get("score", parsed.get("note", 0)))
    try:
        relevance_score = float(relevance_score)
    except (TypeError, ValueError):
        relevance_score = 0.0

    return {
        "relevance_score": relevance_score,
        "relevant": bool(parsed.get("relevant", relevance_score >= 6)),
        "explanation": clean_value(parsed.get("explanation")),
        "answer": clean_value(parsed.get("answer")),
        "reformulated_query": clean_value(parsed.get("reformulated_query")),
    }


def strip_markdown_json_fence(raw_response):
    text = str(raw_response or "").strip()
    if not text.startswith("```"):
        return text

    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def build_llm_json_fallback(raw_response):
    return {
        "relevance_score": 0.0,
        "relevant": False,
        "explanation": str(raw_response or "").strip(),
        "answer": "",
        "reformulated_query": "",
    }


def make_reformulation_unique(reformulated_query, attempted_queries, fallback_seed, attempt_number, clean_value):
    normalized_attempts = {normalize_query_for_comparison(query, clean_value) for query in attempted_queries}
    if normalize_query_for_comparison(reformulated_query, clean_value) not in normalized_attempts:
        return reformulated_query

    suffixes = [
        "patent invention technical system",
        "apparatus method implementation patent",
        "technical abstract claims invention",
    ]
    suffix = suffixes[(attempt_number - 1) % len(suffixes)]
    return f"{fallback_seed.strip()} {suffix}"


def normalize_query_for_comparison(query, clean_value):
    return " ".join(clean_value(query).lower().split())


def truncate_text(value, max_length, clean_value):
    text = clean_value(value)
    if len(text) <= max_length:
        return text
    return f"{text[:max_length].rstrip()}..."
