"""Testes do subcomando de execução remota do CLI."""

import importlib.util
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_cli():
    spec = importlib.util.spec_from_loader(
        "pdc_cli", importlib.machinery.SourceFileLoader("pdc_cli", str(BASE_DIR / "pdc")))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import importlib.machinery  # noqa: E402

cli = _load_cli()

PRONTO = {
    "arch": "aarch64", "python": "3.11", "venv": "sim", "docker": "27.3.1",
    "compose": "sim", "docker_sem_sudo": "sim", "rsync": "sim", "repo": "presente",
    "git": "2.47.0", "origin": "alcancavel",
}


class ProbeScriptTest(unittest.TestCase):
    """O script do probe roda em qualquer shell POSIX e produz chave=valor."""

    def test_saida_parseavel_nesta_maquina(self):
        result = subprocess.run(
            ["bash", "-s", "--", str(BASE_DIR)],
            input=cli.REMOTE_CHECK_SCRIPT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        facts = cli._parse_facts(result.stdout)
        for key, _label in cli.REMOTE_CHECK_FIELDS:
            self.assertIn(key, facts, f"probe não reportou {key}")
        self.assertEqual(facts["arch"], os.uname().machine)
        # O repositório é este; o probe tem de enxergar o ./pdc.
        self.assertEqual(facts["repo"], "presente")

    def test_repo_ausente_quando_caminho_nao_tem_pdc(self):
        result = subprocess.run(
            ["bash", "-s", "--", "/tmp"],
            input=cli.REMOTE_CHECK_SCRIPT, text=True, capture_output=True)
        self.assertEqual(cli._parse_facts(result.stdout)["repo"], "ausente")


class ResolveCaminhoTest(unittest.TestCase):
    """O ~ não expande dentro de "$1"; o prelúdio resolve isso no device."""

    def _resolve(self, argumento):
        result = subprocess.run(
            ["bash", "-s", "--", argumento],
            input=cli.REMOTE_RESOLVE + 'echo "$P"\n',
            text=True, capture_output=True, env={**os.environ, "HOME": "/home/teste"})
        return result.stdout.strip()

    def test_til_vira_home(self):
        self.assertEqual(self._resolve("~/pdc-ia"), "/home/teste/pdc-ia")

    def test_relativo_vira_home(self):
        self.assertEqual(self._resolve("pdc-ia"), "/home/teste/pdc-ia")

    def test_absoluto_fica_intacto(self):
        self.assertEqual(self._resolve("/opt/pdc"), "/opt/pdc")


class BlockersTest(unittest.TestCase):
    def test_device_pronto_nao_tem_pendencia(self):
        self.assertEqual(cli._remote_blockers(PRONTO), [])

    def test_docker_ausente_e_pendencia(self):
        blockers = cli._remote_blockers({**PRONTO, "docker": "AUSENTE"})
        self.assertTrue(any("Docker não instalado" in b for b in blockers))

    def test_docker_exigindo_sudo_e_pendencia(self):
        blockers = cli._remote_blockers({**PRONTO, "docker_sem_sudo": "NAO"})
        self.assertTrue(any("usermod -aG docker" in b for b in blockers))

    def test_python_antigo_e_pendencia(self):
        blockers = cli._remote_blockers({**PRONTO, "python": "3.9"})
        self.assertTrue(any("3.10+" in b for b in blockers))

    def test_versao_ilegivel_nao_explode(self):
        blockers = cli._remote_blockers({**PRONTO, "python": "erro"})
        self.assertTrue(any("versão do Python" in b for b in blockers))

    def test_arm32_bloqueado_por_falta_de_runtime(self):
        blockers = cli._remote_blockers({**PRONTO, "arch": "armv7l"})
        self.assertTrue(any("ARM64" in b for b in blockers))


class ModoDeEntregaTest(unittest.TestCase):
    """No modo git quem baixa é o device; rsync só importa no --from-local."""

    def test_git_ausente_barra_modo_git(self):
        blockers = cli._remote_blockers({**PRONTO, "git": "AUSENTE"}, modo="git")
        self.assertTrue(any("git ausente" in b for b in blockers))

    def test_sem_internet_no_device_sugere_from_local(self):
        blockers = cli._remote_blockers({**PRONTO, "origin": "INALCANCAVEL"}, modo="git")
        self.assertTrue(any("--from-local" in b for b in blockers))

    def test_rsync_nao_e_exigido_no_modo_git(self):
        self.assertEqual(cli._remote_blockers({**PRONTO, "rsync": "NAO"}, modo="git"), [])

    def test_rsync_e_exigido_no_modo_local(self):
        blockers = cli._remote_blockers({**PRONTO, "rsync": "NAO"}, modo="local")
        self.assertTrue(any("rsync ausente" in b for b in blockers))

    def test_git_nao_e_exigido_no_modo_local(self):
        self.assertEqual(cli._remote_blockers({**PRONTO, "git": "AUSENTE"}, modo="local"), [])


class DefasagemTest(unittest.TestCase):
    """O device baixa do GitHub, não desta pasta: divergência tem de avisar."""

    def test_arquivo_sem_commit_gera_aviso(self):
        with mock.patch.object(cli, "_git_local") as g:
            g.side_effect = lambda *a, **k: " M pdc\n M shared/x.py" if a[0] == "status" else "0"
            avisos = cli._avisos_de_defasagem("main")
        self.assertTrue(any("sem commit" in a for a in avisos))

    def test_commit_nao_enviado_gera_aviso(self):
        with mock.patch.object(cli, "_git_local") as g:
            g.side_effect = lambda *a, **k: "" if a[0] == "status" else "3"
            avisos = cli._avisos_de_defasagem("main")
        self.assertTrue(any("não enviados" in a for a in avisos))

    def test_repo_limpo_e_sincronizado_nao_avisa(self):
        with mock.patch.object(cli, "_git_local") as g:
            g.side_effect = lambda *a, **k: "" if a[0] == "status" else "0"
            self.assertEqual(cli._avisos_de_defasagem("main"), [])


class VarianteTest(unittest.TestCase):
    """Não existe imagem CUDA para ARM neste repositório."""

    def test_arm_forca_cpu(self):
        self.assertTrue(cli._remote_cpu_only({"arch": "aarch64"}))

    def test_x86_deixa_autodetectar(self):
        self.assertFalse(cli._remote_cpu_only({"arch": "x86_64"}))


class LancadorTest(unittest.TestCase):
    """O lançador roda solto e sempre registra o código de saída.

    Sem a sentinela, quem acompanha o log espera para sempre. O teste roda o
    mesmo script que vai para o device, só que num bash local.
    """

    def _disparar(self, comandos, destino):
        return subprocess.run(
            ["bash", "-s", "--", str(destino)],
            input=cli._remote_launcher(comandos), text=True, capture_output=True)

    def _esperar_sentinela(self, destino, limite=15.0):
        log = Path(destino) / cli.REMOTE_LOG
        fim = time.time() + limite
        while time.time() < fim:
            if log.is_file() and "__PDC_RC__=" in log.read_text():
                for linha in log.read_text().splitlines():
                    if linha.startswith("__PDC_RC__="):
                        return int(linha.split("=", 1)[1])
            time.sleep(0.2)
        return None

    def test_codigo_de_saida_chega_na_sentinela(self):
        with tempfile.TemporaryDirectory() as destino:
            resultado = self._disparar("echo trabalhando; exit 3", destino)
            self.assertIn("__PDC_STARTED__", resultado.stdout)
            # exit no meio da cadeia não pode pular a sentinela: daí o subshell.
            self.assertEqual(self._esperar_sentinela(destino), 3)
            self.assertIn("trabalhando", (Path(destino) / cli.REMOTE_LOG).read_text())

    def test_sucesso_registra_zero(self):
        with tempfile.TemporaryDirectory() as destino:
            self._disparar("true", destino)
            self.assertEqual(self._esperar_sentinela(destino), 0)

    def test_pid_registrado_para_acompanhar_processo_morto(self):
        with tempfile.TemporaryDirectory() as destino:
            self._disparar("sleep 5", destino)
            pid_file = Path(destino) / cli.REMOTE_PID
            time.sleep(1.5)
            self.assertTrue(pid_file.is_file(), "sem PID não dá para detectar processo morto")
            self.assertTrue(pid_file.read_text().strip().isdigit())

    def test_segunda_execucao_e_barrada(self):
        with tempfile.TemporaryDirectory() as destino:
            self._disparar("sleep 5", destino)
            time.sleep(1.5)
            segunda = self._disparar("echo nao deveria rodar", destino)
            self.assertIn("__PDC_BUSY__", segunda.stdout)


if __name__ == "__main__":
    unittest.main()
