# Evidências atuais — DrenaVision 0.4.0

As métricas têm escopos diferentes e não devem ser misturadas:

| Arquivo | O que comprova |
|---|---|
| `training_manifest.json`, `training_results.csv` | Treino YOLO real, 63 épocas, melhor na 43; hashes e ambiente |
| `preparation_report.json`, `data_manifest.csv` | Base 1107/60/60, exclusões e isolamento por hash/grupo/semelhança |
| `negative_label_quality_audit.json` | Revisão cega dos negativos que continham água ou eram ambíguos |
| `active_profile.json` | Perfil de imagem fixado antes da avaliação adicional; pesos e parâmetros |
| `val_water_presence_review.json` | Presença de água nas 60 imagens de validação, revisada antes da previsão; uma incerta |
| `water_presence_calibration.json` | Duas rodadas de calibração nos mesmos 59 casos claros; a primeira rejeitada, a segunda selecionada |
| `extra_test_manifest.csv`, `extra_test_protocol.json`, `extra_test_blinded_review.json` | Identidades seladas de 119 imagens adicionais e exclusões feitas antes da inferência |
| `independent_comparison.json` | Produto completo: caixas e presença em 60 imagens de regressão + 119 adicionais |
| `baseline_master_test_metrics.json`, `candidate_master_test_metrics.json`, `master_map_comparison.json` | AP do YOLO isolado nas mesmas 60 referências v3, não do produto composto |
| `saved_images_visual_review.json`, `saved_images_and_false_alerts.jpg` | Revisão dos cinco exemplos de fotos já salvos e falsos alertas preservados |
| `product_validation_v3_raw.json`, `product_validation_v3_confirmed.json` | Oito arquivos completos antes/depois da confirmação temporal, com erros mantidos |
| `video_confirmation_scope.json`, `hanoi_video_temporal_proof.json` | Etapa temporal somente em vídeo; sete propostas brutas de água suprimidas, demais previsões idênticas |
| `ui_verification.json`, `frontend_*.jpg` | Envio real de foto/vídeo, reprodução dos 12 s e histórico na interface |
| `water_model_source/` | Manifesto e licença exatos do B1; pesos ficam na release/local |

O perfil de imagem é `pipeline-1314fcfa94f5f8cf`. O B1 é pré-treinado pela
NVIDIA; somente o YOLO foi ajustado neste projeto. As caixas de água derivam
de componentes previstos e podem incluir espaços secos. Pontos de roedores
e anotações de categorias isoladas são referências parciais; uma foto pode
conter água ou outros animais que a fonte não anotou.

No conjunto adicional, água foi encontrada em 33/40 positivas, com dois falsos
alertas nos dez negativos revisados. Animal apareceu em 46/69, contra 47/69
do v2. Nenhuma conclusão de acurácia universal decorre desses números.

Vídeos usam o mesmo perfil-base e uma etapa temporal adicional. Essa etapa
é avaliada nos exemplos conhecidos; não foi avaliada pelas 179 imagens. As
contagens técnicas são marcações/IDs, não vários animais físicos. Veja o
[protocolo](../../../docs/VALIDACAO_V3.md) e a
[ficha do modelo](../../../docs/MODEL_CARD.md).

As capturas/montagens incluem derivados acadêmicos das fontes indicadas nos
manifestos: CLICK e Stagnant Water (CC BY 4.0), iRodent (imagens tratadas como
CC BY-NC), e exemplos Wikimedia com autores/licenças por arquivo em
`dados/examples/sources.json`. Alterações: predição real, caixas, máscaras,
redimensionamento e montagem para inspeção. Atribua também as fontes ao usar
essas imagens. O código AGPL não altera a licença das mídias ou dos pesos.
