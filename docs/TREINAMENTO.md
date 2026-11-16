# Treinamento customizado

## Condições de início

Não inicie treino enquanto qualquer item estiver ausente:

- dataset YOLO com `train`, `val` e `test` não vazios;
- mapa exato das três classes;
- manifesto com fonte, licença, split, grupo e hash;
- validação sem erros, preferencialmente com pHash;
- `configs/train.yaml` revisado;
- espaço em disco e ambiente Python verificados.

```powershell
python scripts/validate_dataset.py `
  --dataset-root datasets/processed/drenavision `
  --yaml datasets/processed/drenavision/dataset.yaml `
  --manifest datasets/manifests/dataset_manifest.csv `
  --phash `
  --report artifacts/dataset_validation.json
```

## Configuração principal

O baseline é um detector nano pré-treinado oficialmente suportado. A
configuração atual usa `yolo26n.pt`; isso atende à preferência por uma variante
leve sem limitar o projeto a uma geração específica.

Parâmetros importantes:

| Parâmetro | Papel | Valor inicial prudente |
|---|---|---|
| `epochs` | máximo de épocas | 100 com early stopping |
| `imgsz` | dimensão de entrada | 640 |
| `batch` | imagens por lote | 8 em GPU de 8 GB; reduzir se faltar VRAM |
| `patience` | early stopping | 20 |
| `seed` | protocolo | 42, fixo |
| `device` | acelerador | `auto` |
| `workers` | carregamento | 4 no Windows; reduzir se instável |
| `close_mosaic` | desativar mosaic no fim | 10 |

Não ajuste hiperparâmetros olhando o conjunto de teste. Use validação para
seleção; teste é reservado à avaliação final.

## Execução

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/train.py --config configs/train.yaml
```

Seleção de dispositivo: CUDA quando disponível, MPS em macOS compatível e CPU
como fallback. Treino em CPU é funcional, mas pode ser muito lento.

## Prova de fine-tuning

Uma execução válida produz `training_manifest.json` com:

- modelo e hash dos pesos iniciais;
- hash do `best.pt`;
- `weights_changed_from_base=true` e `weights_changed_during_invocation=true`;
- fingerprint do dataset e contagem por split;
- configuração efetiva, seed, dispositivo e versões;
- horários UTC e status concluído.

O pipeline carrega novamente `best.pt`, confere o mapa de classes e só então
copia para `artifacts/models/best.pt`. Hash igual ao ponto de partida invalida a
alegação de ajuste.

Na execução desta versão, o treino foi retomado após a época 28, concluiu 70 de
até 100 épocas e parou antecipadamente com `patience=20`; o melhor checkpoint
foi selecionado na época 50. A evidência curada está em
`artifacts/metrics/training_manifest.json` e `training_results.csv`.

## Ajuste por memória

Se ocorrer “CUDA out of memory”:

1. encerre outros processos GPU;
2. reduza `batch` de 8 para 4 ou 2;
3. somente depois reduza `imgsz`;
4. registre a nova configuração;
5. crie uma nova run, sem sobrescrever a anterior.

Não compare runs com configurações/datasets diferentes sem identificá-los.

## Colab

Use `notebooks/drenavision_training.ipynb` quando o ambiente local não tiver
GPU adequada. Salve dataset e artefatos no Drive ou baixe-os antes de a sessão
expirar. O notebook chama os mesmos scripts; não constitui uma implementação
paralela.

## Após o treino

```powershell
Get-FileHash artifacts/models/best.pt -Algorithm SHA256
python scripts/evaluate.py --config configs/evaluate.yaml
```

Arquive o checkpoint fora do Git, publique por release/armazenamento e registre
hash e instruções de obtenção. Não envie pesos grandes ao Git sem Git LFS ou
estratégia equivalente.

