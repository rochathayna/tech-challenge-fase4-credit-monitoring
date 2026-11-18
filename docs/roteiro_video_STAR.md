# Roteiro do vídeo — método STAR (≤ 5 min)

> Dica de gravação: tela do navegador + câmera no canto. Abas para deixar abertas antes de gravar:
> 1. Site do projeto: https://rochathayna.github.io/tech-challenge-fase4-credit-monitoring/
> 2. Dashboard: `…/reports/monitoring_dashboard.html` (link no site)
> 3. Relatório Evidently do mês 6: `…/reports/drift/month_06.html` (link no site)
> 4. GitHub: `src/contracts.py` e `src/simulate_drift.py`
> 5. *(se tiver Docker)* Grafana em http://localhost:3000 com `make grafana` rodando há ~1 min (replay em andamento)
> 6. *(opcional)* MLflow (`make mlflow`) e Evidently UI (`make evidently-ui`), após `make all`

---

## S — Situation (0:00 – 0:45)

**Tela:** README, seção "Contexto".

> "Uma fintech de crédito tem um modelo de credit scoring em produção. O problema é que a economia mudou: inflação, juros altos, um novo perfil de cliente chegando. O modelo continua respondendo, sem erro, sem exceção no log. Só que pode estar decidindo errado. Isso é **degradação silenciosa**: o sistema parece saudável, mas a fintech está aprovando quem não vai pagar. No nosso experimento, a inadimplência entre os clientes **aprovados** saiu de 7 % para 21 % em seis meses, sem nenhum erro técnico."

## T — Task (0:45 – 1:15)

**Tela:** README, tabela "Resultados em uma tela".

> "Nossa tarefa foi construir a camada de sustentação desse modelo com quatro objetivos: **um**, impedir que dado ruim entre no modelo; **dois**, detectar estatisticamente o drift; **três**, dar visibilidade operacional ao pipeline com logs, métricas e alertas; **quatro**, garantir compliance com a LGPD e explicar *por que* o modelo degradou."

## A — Action (1:15 – 3:30)

**1. Contratos de dados (1:15 – 1:55).** Tela: `src/contracts.py` → terminal `make all` (início do log).
> "Usamos o **Pandera** com 10 regras rígidas: sem nulos, idade de 18 a 100, limite de renda, sem duplicatas, domínios fechados e coerência entre renda e comprometimento. Duas regras são de privacidade: o schema é estrito, então se um CPF em claro vazar, o lote é bloqueado. O contrato pegou erros reais no dataset do Kaggle, como gente com 144 anos e 123 anos de emprego, e bloqueou o lote corrompido do parceiro, que foi para a quarentena."

**2. Modelo e simulação do drift (1:55 – 2:35).** Tela: `src/simulate_drift.py` (docstring).
> "Treinamos um XGBoost com AUC de 0,95. Depois simulamos seis meses de produção com quatro mecanismos: mudança no perfil de clientes, inflação, alta de juros, que são *data drift*, e estresse econômico, que é *concept drift*: o mesmo cliente passa a ter mais chance de não pagar."

**3. Detecção e observabilidade (2:35 – 3:30).** Tela: relatório Evidently do mês 6 → Grafana (ou as capturas do Prometheus no site) → MLflow.
> "Para cada mês, o **Evidently** compara produção com referência usando PSI e Kolmogorov-Smirnov. Confirmamos drift só quando há efeito relevante (PSI ≥ 0,1) *e* significância (p < 0,05). Tudo vai para um log estruturado central. Dele saem o **MLflow**, com um run por lote, o **Evidently UI** com o histórico, e um exporter para o **Prometheus**, que tem 7 regras de alerta, como queda de AUC, dataset drift e lote bloqueado. O **Grafana** mostra a degradação acontecendo mês a mês."

## R — Result (3:30 – 5:00)

**Tela:** site do projeto → dashboard: KPIs → gráfico de AUC → heatmap PSI → tabela de alertas → análise causal.

> "O resultado: o modelo sai de 0,94 para 0,88 de AUC. O primeiro alerta crítico aparece no **mês 3**, dois meses antes de o drift do dataset agregado aparecer. Juros, comprometimento de renda, valor do empréstimo e grade são as features degradadas.
>
> A análise causal foi o maior aprendizado. Ligando um mecanismo por vez, vimos que o **concept drift** é o que mais derruba o modelo e é **invisível** ao monitoramento de features: zero alertas de drift. Já a mudança de perfil gera a maior parte dos alertas sem piorar a AUC. Conclusão: drift não é sinônimo de degradação, e monitorar só features não basta. Tem que acompanhar o score, a calibração e a performance com rótulo.
>
> Na governança, o CPF é pseudonimizado na entrada, dados sensíveis não são usados, a base legal é a proteção ao crédito, temos plano de retenção e revisão humana para decisões automatizadas, como manda o artigo 20 da LGPD. Também medimos viés por idade e moradia.
>
> Lições: contrato de dados é a primeira linha de defesa, métricas de negócio contam a história melhor que AUC, e LGPD pode ser uma regra automatizada no pipeline. Obrigado!"

---

### Checklist
- [ ] ≤ 5 minutos
- [ ] Mostrar o lote sendo **bloqueado** no terminal (linha `batch_blocked`)
- [ ] Mostrar o relatório Evidently com features "Detected"
- [ ] Mostrar o dashboard com alertas
- [ ] Mostrar o site do GitHub Pages (o professor consegue abrir tudo por ele)
- [ ] Mencionar base legal + retenção + art. 20
- [ ] (Opcional) colar o link do vídeo no topo do README
