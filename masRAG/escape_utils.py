def escape_curly_braces(text: str) -> str:
    return text.replace('{', '{{').replace('}', '}}')
