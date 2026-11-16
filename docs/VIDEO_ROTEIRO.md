# Roteiro canônico do vídeo técnico

**Duração-alvo:** 4min30s (faixa editorial de 3 a 5 minutos). Essa duração é
uma escolha do DrenaVision, não um limite oficial de Applied Computer Vision.

Antes da gravação, confirme que existem validação do dataset, manifesto de
treino, `best.pt`, `metrics.json`, imagens reais do split de teste e vídeo com
origem registrada. Sem essas evidências, adie a gravação ou declare o bloqueio.

## Plano de cenas

| Tempo | Cena e tela | Narração sugerida | Comando/arquivo |
|---|---|---|---|
| 0:00-0:30 | README, Ficha Técnica, três classes e arquitetura | “O DrenaVision é o módulo de visão da DrenaHub. Ele sinaliza obstrução visível, água acumulada e animal visível para inspeção humana; não prevê alagamento nem mede profundidade.” | `README.md`, `docs/ARQUITETURA.md` |
| 0:30-1:20 | manifesto de dados, validação, config e hashes de treino | “As fontes têm versão e licença registradas. Os grupos não atravessam splits. Este manifesto mostra ambiente, fingerprint e que os pesos customizados diferem dos iniciais.” | `python scripts/validate_dataset.py --dataset-root datasets/processed/drenavision --yaml datasets/processed/drenavision/dataset.yaml --manifest datasets/manifests/dataset_manifest.csv --phash --report artifacts/dataset_validation.json`; `configs/train.yaml`; `training_manifest.json` |
| 1:20-1:55 | Streamlit: original e detecção de obstrução em imagem de teste | “O exemplo não participou do treino. A confiança pertence à detecção e não é probabilidade de alagamento. O evento recomenda revisão.” | imagem com `split=test`; evidência em `artifacts/events/evidence` |
| 1:55-2:25 | TP real de água e FN predefinido | “Este caso tem ao menos um TP, mas a classe teve recall 0,1159. O exemplo predefinido foi perdido. Sem calibração, não estimamos profundidade, volume ou vazão.” | `artifacts/demo/stagnant_water_78c5a3feac2bba2c_annotated.jpg` e `stagnant_water_04213a44e43bc933_annotated.jpg` |
| 2:25-3:05 | TP real de animal, vídeo, `track_id` e histórico | “O TP foi selecionado após a avaliação apenas para demonstração e ainda contém erro. A sequência predefinida detectou duas obstruções, mas perdeu água e animal; os FNs permanecem visíveis.” | `artifacts/demo/irodent_311988d4ce1851a6_annotated.jpg`; `video_inference_manifest.json`; MP4 da release |
| 3:05-3:40 | Swagger/terminal com JSON real e histórico | “A API retorna evento, ponto, UTC, versão, classes, caixas e confiança. O vídeo também registra frame/timestamp e evidência JSON.” | `curl.exe -X POST http://127.0.0.1:8000/vision/predict -F "point_id=garagem-demo-01" -F "file=@caminho\imagem_teste.jpg;type=image/jpeg"`; `GET /vision/events` |
| 3:40-4:20 | métricas, matriz e TP/FP/FN | “No teste separado: P 0,4447, R 0,4796, mAP50 0,3781 e mAP50-95 0,2196. Água teve recall 0,1159; por isso o sistema só recomenda inspeção.” | `artifacts/metrics/evaluation_metrics.json`, matrizes e `error_examples/`, `docs/MODEL_CARD.md` |
| 4:20-4:30 | interface com disclaimer | “A solução entrega evidência reproduzível com revisão humana. Próximos passos são dados próprios, validação por local e integração com sensores.” | tela final |

Leia apenas valores dos manifests e do JSON da avaliação. Não transforme os
exemplos demonstrativos selecionados em estimativas de desempenho.

## Serviços para a gravação

```powershell
# Terminal 1
.\.venv\Scripts\Activate.ps1
python -m uvicorn drenavision.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2
.\.venv\Scripts\Activate.ps1
python -m streamlit run drenavision/ui/streamlit_app.py
```

Verifique `/health` antes do REC. Ele deve mostrar o modelo carregado. Não use
um detector genérico ou resposta simulada para contornar HTTP 503.

## Arquivos que aparecem na tela

- `datasets/manifests/dataset_manifest.csv`;
- `datasets/manifests/dataset_summary.json`;
- `artifacts/dataset_validation.json`;
- `configs/train.yaml`;
- `artifacts/metrics/training_manifest.json`;
- `artifacts/models/best.pt` (mostrar hash, não abrir o binário);
- `artifacts/metrics/evaluation_metrics.json`;
- matriz de confusão e exemplos de erro;
- imagens/vídeo de teste com origem registrada;
- JSON real da API e histórico.

## Checklist OBS

### Antes de gravar

- [ ] 1920x1080, 30 fps e áudio sem clipping;
- [ ] notificações, e-mail e mensageiros fechados;
- [ ] zoom entre 125% e 150% e fonte do terminal legível;
- [ ] `.env`, tokens, usuário local e caminhos privados fora da tela;
- [ ] pessoas/placas anonimizadas ou autorizadas;
- [ ] API e Streamlit iniciados;
- [ ] `/health` pronto e checkpoint correto;
- [ ] exemplos confirmados como `test`;
- [ ] métricas conferidas diretamente no JSON;
- [ ] ensaio cronometrado.

### Durante

- [ ] mostrar original e anotado;
- [ ] provar fine-tuning com hashes/manifesto;
- [ ] executar inferência real;
- [ ] exibir JSON e histórico;
- [ ] mostrar um erro real, não só acertos;
- [ ] declarar limites de água e animal;
- [ ] não usar corte para fazer simulação parecer execução.

### Depois

- [ ] conferir áudio, legibilidade e duração;
- [ ] remover segredo ou dado pessoal que escapou;
- [ ] comparar todos os números com `metrics.json`;
- [ ] adicionar créditos das imagens;
- [ ] exportar MP4 H.264;
- [ ] publicar sem exigir login;
- [ ] testar o link em janela anônima e outro dispositivo;
- [ ] registrar URL no README e relatório posterior;
- [ ] manter o link público durante a correção.

O vídeo de entrada local para testar a interface é
`artifacts/videos/demo_input_test_sequence.mp4`. Use a entrada sem anotações;
`drenavision_demo_annotated.mp4` já é o resultado processado. Se o arquivo de
entrada não estiver no clone, baixe-o pela
[release `v0.1.0-model`](https://github.com/Torugo0/drenahub-drenavision/releases/tag/v0.1.0-model).

