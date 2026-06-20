# Dados

## `raw/credit.csv` — base sintética de risco de crédito (demonstração)

**Fonte:** dados **sintéticos**, gerados pelo script reprodutível
[`scripts/generate_demo_dataset.py`](../scripts/generate_demo_dataset.py).
Nenhum dado pessoal real é utilizado (atende RNF07). Para regenerar:

```bash
python -m scripts.generate_demo_dataset            # gera data/raw/credit.csv (seed=42)
python -m scripts.generate_demo_dataset --rows 2000 --seed 7
```

**Tarefa:** classificação binária da variável-alvo `default` (1 = inadimplência).

**Dimensões:** ~1005 linhas (1000 base + 5 duplicatas exatas) × 11 colunas.

### Dicionário de dados

| Coluna              | Tipo        | Descrição                                              |
|---------------------|-------------|--------------------------------------------------------|
| `customer_id`       | inteiro     | Identificador do cliente (coluna de id, não-preditiva). |
| `age`               | inteiro     | Idade em anos (18–75).                                  |
| `income`            | numérico    | Renda anual estimada (~6% faltante).                    |
| `employment_length` | numérico    | Anos de emprego (0–40, ~6% faltante).                   |
| `loan_amount`       | numérico    | Valor do empréstimo solicitado.                         |
| `debt_to_income`    | numérico    | Razão dívida/renda anual (≈ 0,1–0,6 típico).            |
| `num_credit_lines`  | inteiro     | Número de linhas de crédito abertas.                    |
| `home_ownership`    | categórica  | `rent` \| `mortgage` \| `own`.                          |
| `purpose`           | categórica  | Finalidade do empréstimo; `small_business` é rara (~1%).|
| `region`            | categórica  | `north` \| `south` \| `east` \| `west`.                 |
| `default`           | alvo (0/1)  | Inadimplência. Classe positiva ≈ 23% (desbalanceada).   |

### Características intencionais (exercitam o pipeline)

- **Desbalanceamento** (~23% positivos) → torna `macro_f1` mais informativa que a acurácia.
- **Valores faltantes** em `income` e `employment_length` → imputação reprodutível.
- **Duplicatas exatas** → remoção controlada pelo Agente de Qualidade e Limpeza.
- **Categoria rara** (`purpose = small_business`) → tratamento de categorias raras.
- **Relação alvo↔features** logística + ruído, determinística por `seed` → sinal aprendível e reprodutível.

> A política de `.gitignore` ignora `data/raw/*` por padrão (dados sensíveis não
> devem ser versionados). Esta base sintética é uma exceção explícita por ser
> pública, anônima e necessária para a demonstração de ponta a ponta.
