import os


def get_local_env(name: str, default: str = "") -> str:
    """Read an env var from the current process, then Windows user/machine env."""
    value = os.getenv(name, "").strip()
    if value:
        return value

    if os.name != "nt":
        return default

    value = _get_windows_env_var(name, scope="user").strip()
    if value:
        return value

    value = _get_windows_env_var(name, scope="machine").strip()
    if value:
        return value

    return default


def _get_windows_env_var(name: str, scope: str) -> str:
    try:
        import winreg
    except ImportError:
        return ""

    if scope == "user":
        root = winreg.HKEY_CURRENT_USER
        subkey = "Environment"
    elif scope == "machine":
        root = winreg.HKEY_LOCAL_MACHINE
        subkey = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
    else:
        return ""

    try:
        with winreg.OpenKey(root, subkey) as key:
            value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return ""
    return str(value)
