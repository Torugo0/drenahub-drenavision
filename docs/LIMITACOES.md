# Limitações essenciais

- o sistema não prevê alagamento nem estima profundidade, vazão ou volume;
- `animal_visivel` não significa obstrução e não identifica espécie;
- a fonte animal é concentrada em roedores;
- reflexos/piso molhado podem confundir `agua_acumulada`;
- resíduos variados e oclusão dificultam `obstrucao_visivel`;
- zero detecções não comprova segurança;
- confiança não é probabilidade calibrada de risco;
- tracking pode trocar IDs e não é censo de animais;
- domínio, câmera, iluminação e estação podem reduzir desempenho;
- datasets públicos não substituem validação em ambientes da DrenaHub;
- imagens podem conter dados pessoais e exigem governança/LGPD;
- iRodent é tratado como CC BY-NC, impedindo presumir uso comercial;
- Ultralytics AGPL-3.0 exige decisão de licença antes de produto fechado;
- revisão humana e inspeções periódicas independentes continuam obrigatórias.

Uma análise completa de riscos, mitigação, privacidade e condições para piloto
está em [ETHICS_AND_LIMITATIONS.md](ETHICS_AND_LIMITATIONS.md).

