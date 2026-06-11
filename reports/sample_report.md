# Relatório Técnico — (exemplo)

> Modelo de relatório final gerado automaticamente pelo Agente Relator e Auditor.
> Os campos abaixo seguem a estrutura esperada na seção 24 do documento de apoio.

## 1. Resumo executivo
Descrição do problema e da base utilizada.

## 2. Definição da tarefa
- **Variável-alvo:** `<target>`
- **Tipo de tarefa:** classification | regression | anomaly
- **Unidade de observação:** `<...>`
- **Métrica primária:** `<macro_f1 | rmse | ...>`
- **Critério de sucesso:** `<...>`

## 3. Perfil dos dados
- Linhas / colunas: `<n_rows>` × `<n_cols>`
- Tipos inferidos, faltantes, duplicatas e desbalanceamento.

## 4. Limpeza e engenharia de atributos
Ações aplicadas e justificativas (imputação, duplicatas, categorias raras, transformações).

## 5. Estratégia de validação
Esquema de validação cruzada adotado e medidas de prevenção de *leakage*.

## 6. Modelos testados
Famílias avaliadas e hiperparâmetros principais.

## 7. Resultados por métrica e por fold
| Modelo | Métrica primária (média) | Desvio | Folds | Tempo (s) |
|--------|--------------------------|--------|-------|-----------|
| ...    | ...                      | ...    | ...   | ...       |

## 8. Autoencoder vs. baseline
Aplicação escolhida (representação / denoising / anomalia) e comparação contra a linha de base.

## 9. Modelo recomendado
Modelo selecionado, justificativa do critério e limitações.

## 10. Riscos, vieses e restrições de uso
Considerações éticas, vieses observados e restrições do dataset.

## 11. Próximos passos
Mais dados, novas features, ajuste de modelos, monitoramento ou retreinamento.

---
*Reprodutibilidade:* seed, versões de bibliotecas, versão dos dados e configuração ficam registrados no PostgreSQL.
