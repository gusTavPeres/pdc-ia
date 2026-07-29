# Modelos

O catálogo fica em `config/models.yaml`. Os artefatos baixados ficam em
`data/models/`, fora do Git, e são compartilhados pelos três serviços.

## Cadastro Pelo Dashboard

Abra `Configuração > Modelos`. Para o cadastro básico, selecione o tipo,
preencha a fonte e clique em `Salvar Modelo`:

| Tipo | Fonte esperada |
| --- | --- |
| Vision | link direto para um arquivo `.pt` compatível com Ultralytics |
| SLM | link direto para um arquivo `.gguf` |
| Audio ASR | identificador aceito pelo `faster-whisper`, como `small` ou `large-v3` |
| Audio TTS | link do arquivo Piper `.onnx` e, se necessário, do `.onnx.json` |

Links do Hugging Face com `/blob/` ou `/raw/` são convertidos para
`/resolve/`. Links de releases do GitHub também funcionam quando apontam para o
arquivo, não para a página da release. A plataforma não converte páginas de
modelo, repositórios completos, arquivos Ollama nem formatos diferentes dos
indicados acima.

No TTS, o segundo link pode ficar vazio quando o arquivo de configuração usa o
nome convencional `<modelo>.onnx.json`; nesse caso, a interface completa o
endereço automaticamente.

As opções principais têm estes efeitos:

- `Nome amigável`: texto exibido no catálogo;
- `Tornar padrão`: usa o modelo como seleção inicial em Benchmark e Inferência;
- `Carregar agora`: solicita a troca no container correspondente após salvar.

Salvar a definição não garante que o modelo seja compatível. A verificação
real ocorre no download ou no carregamento pelo runtime.

## Opções Avançadas

- `Chave técnica`: identificador estável usado no YAML, na API e nos resultados;
- `Arquivo local`: nome do artefato em `data/models/`;
- `Tamanho` e `RAM mínima`: metadados informativos do catálogo;
- `GPU layers`: número de camadas GGUF solicitado ao `llama-cpp-python`;
- `Contexto`: janela máxima solicitada ao runtime SLM;
- `Formato de chat`: template usado na montagem do prompt;
- `SHA-256`: valida o artefato principal antes de aceitá-lo no cache.

Os valores informados em `Tamanho`, `RAM mínima` e `GPU layers` não reservam
recursos nem provam compatibilidade com o equipamento.

## Download Pelo CLI

```bash
./pdc models list
./pdc models download defaults
./pdc models download vision yolov8n
./pdc models download slm tinyllama-1.1b-q4
./pdc models download audio_tts --defaults
```

`defaults` baixa os artefatos diretos dos modelos padrão. O comando abaixo
baixa todos os artefatos diretos cadastrados e pode consumir vários gigabytes:

```bash
./pdc models download all
```

Modelos ASR não têm URL direta no catálogo. O `faster-whisper` baixa seus
arquivos na primeira carga. Downloads interrompidos usam arquivos temporários e
o cache só é aceito quando o artefato final não está vazio e, quando informado,
o SHA-256 confere.

## Ativação e Cache

O catálogo é persistido imediatamente no host. `Carregar agora` exige que o
container da frente esteja saudável e pode iniciar um download demorado. Sem
essa opção, o novo padrão será usado na próxima inicialização ou ativação.

Remover um modelo pelo dashboard apaga somente sua definição no catálogo. O
arquivo já baixado permanece em `data/models/`; isso evita novo download caso a
entrada seja recriada. A limpeza desse diretório é uma operação manual.

## Edição Manual

O dashboard cobre o cadastro normal. Para revisão em lote, o mesmo catálogo
pode ser editado em `config/models.yaml`:

```yaml
slm:
  default: tinyllama-1.1b-q4
  models:
    tinyllama-1.1b-q4:
      format: gguf
      url: https://huggingface.co/.../resolve/main/model.gguf
      size_mb: 637
      min_ram_gb: 2
      gpu_capable: true
      gpu_layers: 33
      context_length: 2048
      chat_format: tinyllama
      description: TinyLlama 1.1B Q4
```

Depois de uma edição manual, atualize a página para recarregar o catálogo. Um
serviço que já carregou outro modelo só muda ao receber uma ativação ou ao ser
reiniciado.
