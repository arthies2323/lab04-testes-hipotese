**Pergunta:** o tempo até o merge de PRs difere entre os grupos com e sem participação do bot Copilot no repositório `dotnet/runtime`?

**Welch:** H₀: μ_com_bot = μ_sem_bot; H₁: μ_com_bot ≠ μ_sem_bot, para as médias em horas.
**Mann-Whitney:** H₀: as distribuições são iguais; H₁: há diferença captável pelos postos. O teste não é um teste geral de qualquer diferença de forma nem, sem pressupostos adicionais, de igualdade de medianas.
Os dois testes são **bicaudais**, com α = 0,05, pois uma diferença pode ocorrer em qualquer direção. Os alvos dos testes são distintos; não se deve tratar seus p-valores como respostas idênticas sobre médias.

**Origem:** `data/prs.csv`, amostra já fornecida de 60 PRs (30 por grupo) do `dotnet/runtime`. Segundo `data/README.md`, a coleta ocorreu em 04/11/2026 via `gh pr list`, entre 600 PRs mergeados recentes, com amostragem de semente 42. O grupo com_bot indica review/comentário de `copilot-pull-request-reviewer`; sem_bot indica ausência desse registro, e não ausência de todo tipo de automação. A origem é a documentada no projeto: esta implementação não refez nem verificou a coleta pela API. Cada PR pode ser consultado em `https://github.com/dotnet/runtime/pull/<pr>`.

**Geração:** para cada grupo g, X_g ~ Lognormal(log(m_g), σ_g), com m_g igual à mediana amostral das horas e σ_g igual ao desvio-padrão amostral de log(horas), ddof=1. A tabela de calibração é calculada do CSV em cada execução, sem parâmetros manuais desatualizados. São geradas 30 observações por grupo, na ordem com_bot e sem_bot, com `numpy.random.default_rng(42)`; o mesmo conjunto é usado em G1, nos testes e nas áreas de p-valor de G2. O exemplo preserva parâmetros diferentes por grupo; somente o histograma nulo de G2 usa parâmetros comuns e semente 43.

**Expectativa anterior à execução:** a assimetria à direita e os valores extremos dos tempos motivam esperar que Mann-Whitney seja menos sensível às caudas do que Welch, quando a diferença for de localização com formas semelhantes. Welch continua sendo o teste adequado se o alvo for especificamente a média; não exige variâncias iguais. A superioridade de um teste não pode ser estabelecida por um único exemplo: a expectativa fica registrada como hipótese para as próximas sprints.

**Limitações:** dados observacionais de um único repositório; participação do bot não aleatória e potencial confusão com tamanho, autor e complexidade do PR. A simulação assume observações independentes, tempos positivos e um modelo lognormal; PRs do mesmo autor podem violar independência. Significância não demonstra causalidade, e ausência de significância não prova equivalência.

**Referências técnicas:** [Welch no SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html) e [Mann-Whitney no SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html). O G2 usa `TtestResult.df` (Satterthwaite) e `method="asymptotic"`, com correção de continuidade, respectivamente.
