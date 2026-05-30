from agents import env


def test_get_local_env_prefers_process_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "process-key")
    monkeypatch.setattr(env, "_get_windows_env_var", lambda name, scope: "registry-key")

    assert env.get_local_env("DEEPSEEK_API_KEY") == "process-key"


def test_get_local_env_falls_back_to_windows_user_env(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(env.os, "name", "nt", raising=False)

    def fake_registry(name, scope):
        return "user-key" if scope == "user" else "machine-key"

    monkeypatch.setattr(env, "_get_windows_env_var", fake_registry)

    assert env.get_local_env("DEEPSEEK_API_KEY") == "user-key"


def test_get_local_env_falls_back_to_windows_machine_env(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(env.os, "name", "nt", raising=False)

    def fake_registry(name, scope):
        return "" if scope == "user" else "machine-key"

    monkeypatch.setattr(env, "_get_windows_env_var", fake_registry)

    assert env.get_local_env("DEEPSEEK_API_KEY") == "machine-key"
