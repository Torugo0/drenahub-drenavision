# Arquitetura - visão operacional

O fluxo canônico é:

```text
download licenciado -> conversão YOLO -> validação -> treino -> teste
                                                            |
                                                            v
                                                     best.pt validado
                                                      /           \
                                               FastAPI          vídeo
                                                  |
                                        Streamlit + SQLite
```

Princípios:

1. **falha fechada:** sem checkpoint customizado não há predição;
2. **um contrato:** UI e integrações usam a mesma camada de serviço;
3. **rastreabilidade:** cada run vincula configuração, dataset e pesos por hash;
4. **separação:** dados/runs não entram no Git; código/manifests entram;
5. **mínimo privilégio:** uploads são limitados e caminhos internos não saem;
6. **humano no circuito:** detecção recomenda inspeção, não toma decisão final.

| Camada | Implementação |
|---|---|
| dados | `app/data`, `scripts/download_datasets.py`, conversão e validação |
| ML | `app/inference`, configs e scripts de treino/avaliação/predição |
| serviço | `app/core` |
| HTTP | `app/api/main.py` |
| UI | `app/ui/streamlit_app.py` |
| estado | SQLite + evidências locais em `artifacts/` |

Para contratos, sequência de imagem, estados de erro, configuração e decisões
de produção, leia [ARCHITECTURE.md](ARCHITECTURE.md).

