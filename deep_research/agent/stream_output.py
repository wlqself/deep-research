from collections.abc import Mapping

from ..citations import append_verified_sources

# 来源引用后缀提取
def verified_answer_suffix(
    raw_answer: str,
    sources: Mapping[str, object],
) -> str:
    # 追加来源引用
    verified_answer = append_verified_sources(raw_answer, sources)
    base_answer = raw_answer.rstrip()

    if not verified_answer.startswith(base_answer):
        return ""

    return verified_answer[len(base_answer):]
