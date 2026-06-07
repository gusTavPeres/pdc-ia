# Modelos

Os modelos ficam fora do Git e são armazenados em `data/models/`.

## Comandos

```bash
./pdc models list
./pdc models download defaults
./pdc models download all
./pdc models download vision yolov8n
./pdc models download slm tinyllama-1.1b-q4
./pdc models download audio_tts --defaults
```

`defaults` baixa apenas os modelos padrão. `all` baixa todos os modelos que possuem URL direta.

## Cadastro Pelo Dashboard

Acesse `Configuração > Modelos`.

Campos principais:

- `Tipo`: frente em que o modelo será usado.
- `Nome amigável`: texto exibido nas listas.
- `Link principal`: URL do arquivo do modelo ou ID do ASR.
- `Tornar padrão`: usa esse modelo como seleção inicial.
- `Carregar agora`: tenta ativar o modelo após salvar.

Campos avançados:

- `Chave técnica`: identificador usado no YAML e nas APIs.
- `Arquivo local`: nome esperado no cache.
- `Tamanho` e `RAM mínima`: metadados usados para seleção e leitura operacional.
- `GPU layers` e `Contexto`: usados por SLM/GGUF.
- `Formato de chat`: template do prompt em SLM.

## Fontes Suportadas

Vision:

- URL direta para `.pt`.
- Hugging Face ou GitHub Releases desde que o link aponte para o arquivo.

SLM:

- URL direta para `.gguf`.
- Links Hugging Face com `/blob/` são convertidos para `/resolve/`.

Audio ASR:

- `model_id` do `faster-whisper`, como `small`, `medium`, `large-v3`.
- O runtime baixa o modelo no primeiro carregamento.

Audio TTS:

- URL do `.onnx`.
- URL do `.onnx.json`; se ausente, o sistema tenta usar o mesmo link com `.json`.

Ollama ainda não é importado diretamente. Para SLM, use o arquivo GGUF.

## Arquivo de Configuração

`config/models.yaml` tem esta estrutura:

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

Ao alterar o YAML manualmente, recarregue o catálogo no dashboard ou reinicie a frente afetada.
