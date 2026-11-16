# Reprodutibilidade e rastreabilidade

## Princípio

Uma execução reproduzível precisa vincular **código, dados, configuração,
ambiente, pesos e resultado**. Ter um `best.pt` sem origem documentada não
comprova treinamento customizado; ter um gráfico sem `metrics.json` e split
identificado não comprova avaliação.

## Ambiente de referência

- Python: 3.12.x;
- PyTorch: 2.14.1;
- torchvision: 0.29.1;
- Ultralytics: 8.4.215;
- OpenCV: 4.14.0.94;
- seed do protocolo: 42.

As dependências de topo estão em `requirements.txt`; as de desenvolvimento em
`requirements-dev.txt`. Registre também sistema operacional, driver, GPU,
`torch.version.cuda` e a saída completa de `pip freeze` da execução entregue.

```powershell
python --version
python -m pip freeze | Out-File -Encoding utf8 artifacts/environment-pip-freeze.txt
nvidia-smi | Out-File -Encoding utf8 artifacts/environment-nvidia-smi.txt
python -c "import torch; print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_runtime': torch.version.cuda, 'device_count': torch.cuda.device_count()})"
```

Não versione segredos, caminhos privados ou um dump indiscriminado de variáveis
de ambiente.

## Protocolo de dados

### Fontes versionadas

| Chave | Versão | Uso | Licença conservadora |
|---|---:|---|---|
| `click_trash` | 2 | obstrução visível | CC BY 4.0 |
| `stagnant_water` | 4 | água acumulada; somente classe source `water` | CC BY 4.0 |
| `irodent` | 1.0 | animal visível | CC BY-NC por cautela |

URLs, checksums, observações e transformações vivem no código e no manifesto.
O downloader usa seed 42 e, para CLICK, seleciona 100/20/20 pares para
train/val/test via HTTP Range. O conversor seleciona dados de cada fonte por
split e escreve `datasets/manifests/dataset_manifest.csv`.

### Aquisição

```powershell
python scripts/download_datasets.py --source all --seed 42
```

Não use `--force` sem registrar por que a fonte precisou ser baixada novamente.
Se um hash esperado falhar, pare: não ajuste o valor no código antes de
confirmar uma nova versão oficial da fonte.

### Conversão

```powershell
python scripts/convert_annotations.py `
  --raw-root datasets/raw `
  --output-root datasets/processed/drenavision `
  --manifest datasets/manifests/dataset_manifest.csv `
  --seed 42
```

Arquivos recebem nomes derivados do hash da imagem. O manifesto registra split,
fonte/versão/licença, `source_group`, SHA-256, dimensões e contagem por classe.

### Validação

```powershell
python scripts/validate_dataset.py `
  --dataset-root datasets/processed/drenavision `
  --yaml datasets/processed/drenavision/dataset.yaml `
  --manifest datasets/manifests/dataset_manifest.csv `
  --phash `
  --phash-threshold 4 `
  --report artifacts/dataset_validation.json
```

O comando precisa terminar com código 0. O `--phash` é mais lento, mas deve ser
usado na execução final para encontrar versões visualmente próximas em splits
diferentes. Alertas exigem revisão; erros impedem treino.

### Imutabilidade de teste

Depois da primeira avaliação usada para tomar decisões, o split de teste não
deve orientar hiperparâmetros ou correções de anotação. Se isso ocorrer,
versione um novo dataset e reserve novo teste. Exemplos escolhidos para o vídeo
devem continuar fora do treino.

## Protocolo de treino

1. versionar `configs/train.yaml`;
2. confirmar relatório de dataset válido;
3. executar no commit que será entregue;
4. não reutilizar diretório de run existente;
5. preservar o manifesto de treino e o `best.pt` original;
6. copiar o checkpoint publicado somente após validar classes e hash.

```powershell
git rev-parse HEAD
python scripts/train.py --config configs/train.yaml
```

O treino faz preflight dos splits, seleciona CUDA > MPS > CPU quando `device` é
`auto` e registra:

- início/fim em UTC;
- modelo base e SHA-256 dos pesos iniciais, quando acessível;
- SHA-256 do melhor checkpoint e do checkpoint publicado;
- confirmação de que os pesos mudaram;
- fingerprint do dataset e contagens por split;
- argumentos efetivos, seed e dispositivo;
- versões do ambiente;
- métricas finais expostas pelo treinador.

Saídas esperadas dentro de `artifacts/training/<run>/`:

```text
effective_config.yaml
training_manifest.json
weights/best.pt
weights/last.pt
... gráficos e logs do Ultralytics
```

O checkpoint publicado em `artifacts/models/best.pt` deve ter o mesmo hash do
`best.pt` registrado. Esses arquivos são ignorados pelo Git; distribua-os por
release/armazenamento apropriado e publique o hash na documentação.

## Protocolo de avaliação

```powershell
python scripts/evaluate.py --config configs/evaluate.yaml
```

A avaliação:

- exige checkpoint customizado;
- valida mapa exato das três classes;
- usa `split=test` explicitamente;
- calcula precision, recall, mAP@50 e mAP@50-95 globais e por classe;
- registra tempos por imagem fornecidos pelo validador;
- exporta inventário de plots e análise de FP/FN;
- grava hash do checkpoint, fingerprint do dataset, argumentos e ambiente.

O arquivo autoritativo é:

```text
artifacts/evaluation/<run>/metrics.json
```

Copie números para README, model card, PDF ou vídeo somente a partir desse
arquivo e mantenha o hash associado. “Melhor resultado observado” sem protocolo
pré-definido incentiva seleção oportunista; prefira uma configuração principal
declarada antes do teste.

## Inferência reproduzível

```powershell
python scripts/predict.py `
  --input datasets/processed/drenavision/images/test/<arquivo>.jpg `
  --model artifacts/models/best.pt `
  --output-dir artifacts/predictions `
  --conf 0.35 `
  --iou 0.45 `
  --device auto
```

Para cada exemplo entregue, registre:

- caminho/ID da imagem no manifesto;
- split `test`;
- hash da imagem;
- hash do checkpoint;
- limiares e `imgsz`;
- dispositivo;
- JSON e imagem anotada gerados.

## Vídeo

```powershell
python scripts/process_video.py `
  --input caminho\entrada.mp4 `
  --output artifacts/predictions/demo.mp4 `
  --model artifacts/models/best.pt `
  --track
```

Guarde o vídeo de entrada, seu hash, resolução/FPS, comando e resumo de tracking.
Se houver transcodificação por codec, registre ferramenta e argumentos.

## API e histórico

Para uma demonstração limpa, use banco e diretório de evidências dedicados:

```powershell
$env:DRENAVISION_DATABASE_PATH = "artifacts/demo/events.db"
$env:DRENAVISION_EVIDENCE_DIR = "artifacts/demo/evidence"
$env:DRENAVISION_MODEL_PATH = "artifacts/models/best.pt"
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Não apague silenciosamente um histórico usado como evidência. Se reiniciar a
demonstração, arquive o banco anterior e registre a ação.

## Notebook Colab

`notebooks/drenavision_training.ipynb` chama os mesmos scripts do repositório;
ele não reimplementa treino em células opacas. O usuário deve preencher a URL
do repositório, escolher armazenamento de dados e salvar artefatos fora da
sessão efêmera. Células não contêm outputs de treino fabricados.

## Checklist de uma execução citável

- [ ] commit SHA registrado;
- [ ] `pip freeze`, hardware e CUDA registrados;
- [ ] URLs/versões/licenças das fontes preservadas;
- [ ] checksums de downloads conferidos;
- [ ] manifesto e resumo do dataset gerados;
- [ ] validação completa, incluindo pHash, passou;
- [ ] config efetiva e seed preservadas;
- [ ] hashes inicial/final comprovam fine-tuning;
- [ ] checkpoint contém mapa exato de classes;
- [ ] avaliação usa teste separado;
- [ ] `metrics.json`, matriz e exemplos de erro arquivados;
- [ ] exemplos do vídeo identificados e fora do treino;
- [ ] nenhuma métrica foi transcrita manualmente sem conferência;
- [ ] limitações e eventuais falhas por classe foram publicadas.

