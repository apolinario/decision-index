# Nadia-1 Small

Modelo desenvolvido por Bernardo B. Perobeli, da Artinos. Submissão da suíte pública textual 0.3: **3,61/100**, com 140.178 requisições e `complete: true`. A pontuação pública é corrigida pelo acaso e pela cobertura; não é acurácia global nem Full score.

## Artefatos fixados

- [Pesos e código de inferência](https://huggingface.co/artinos/Nadia-1-140M/tree/47bf26321b31b84b00507c9fddcecdd3822bab41), revisão `47bf26321b31b84b00507c9fddcecdd3822bab41`. O repositório permanece privado; o acesso dos mantenedores ainda precisa ser combinado para a validação privada.
- [Resultados](https://huggingface.co/datasets/artinos/Nadia-1-Decision-Index/tree/e9a15bb06d8135902fe4d8f125bb679efeb88e53), revisão `e9a15bb06d8135902fe4d8f125bb679efeb88e53`.
- [scores.json](https://huggingface.co/datasets/artinos/Nadia-1-Decision-Index/blob/e9a15bb06d8135902fe4d8f125bb679efeb88e53/runs/nadia-1-small/scores.json), [previsões](https://huggingface.co/datasets/artinos/Nadia-1-Decision-Index/resolve/e9a15bb06d8135902fe4d8f125bb679efeb88e53/runs/nadia-1-small/results.jsonl.gz), [ambiente](https://huggingface.co/datasets/artinos/Nadia-1-Decision-Index/blob/e9a15bb06d8135902fe4d8f125bb679efeb88e53/runs/nadia-1-small/environment.json) e [manifesto de hashes](https://huggingface.co/datasets/artinos/Nadia-1-Decision-Index/blob/e9a15bb06d8135902fe4d8f125bb679efeb88e53/manifesto.json).
- Código GitHub: [betitanx/Nadia-1](https://github.com/betitanx/Nadia-1/tree/06e4319cd8354f3da2d245dfb604dcc430ce7f57), revisão `06e4319cd8354f3da2d245dfb604dcc430ce7f57`, também privado.
- Kit utilizado: `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec`.
- Adaptador: `avaliacao.engine_decision_index:NadiaEngine`, incluído no repositório do modelo.
- SHA-256 do checkpoint: `7ea3e7b6a582a09fcf23439575f8150404c8cf71ac087ddd3dbaab06153dcff1`.

## Configuração de inferência

- NVIDIA L40S; um processo; quatro threads CPU.
- PyTorch `2.11.0+cu128`, Transformers `5.3.0`; FP32, atenção MATH e TF32 desativado.
- 2.048 tokens por campo, incluindo estado, instrução e todas as opções. Até 255 opções por campo.
- Sem truncamento, filtragem ou redução das opções. Se qualquer campo exceder o contexto, toda a requisição é recusada.
- As sete tarefas adicionais (MMLU-Pro, BBH, When2Call, New Yorker, RAGTruth, HoVer e PhishNChips) usaram um campo por lote.
- As demais tarefas usaram até oito campos da mesma requisição por lote, respeitando 8.192 tokens após padding; não houve agrupamento entre requisições.
- 130.316 respostas e 9.862 recusas de contexto, sem erros ou pendências de inferência.

O modelo usa uma cabeça compartilhada para pontuar alternativas. Não gera cadeia de raciocínio, não consulta ferramentas externas e não faz tentativas múltiplas. Para decisões booleanas, devolve a probabilidade de `true`; para escolhas, devolve a distribuição sobre todas as alternativas.

## Reprodução

Após receber acesso ao modelo, baixe a revisão fixada e instale `modelo/requirements.txt` em um ambiente com a versão CUDA indicada. Instale também este kit na revisão fixada. Na raiz do repositório do modelo, o adaptador pode ser usado com:

```bash
python -m decision_index pipeline --edition 0.3 --engine avaliacao.engine_decision_index:NadiaEngine --option repositorio=. --option dispositivo=cuda --option threads=4 --option lote=8 --suite-dir /caminho/suite-0.3 --out runs/nadia-1-small
```

Esse comando faz uma nova execução com limite de lote 8. Para reproduzir exatamente a composição histórica, separe as sete tarefas adicionais e use `--option lote=1` nelas; use lote 8 nas demais. A configuração de cada uma das quatro chamadas está registrada no ambiente. O adaptador foi qualificado em CPU contra o avaliador portátil; os resultados de qualidade submetidos foram obtidos em GPU. A medição oficial de latência deve ser realizada pelos mantenedores em seu hardware.

Para conferir as previsões existentes, não é necessário acesso aos pesos:

```bash
python -m decision_index score --edition 0.3 --suite-dir /caminho/suite-0.3 --results runs/nadia-1-small/results.jsonl.gz --out pontuacao-recalculada
```

## Integridade e limitações

Os três arquivos da suíte e as exclusões passaram na verificação dos hashes oficiais. Foram consolidadas quatro chamadas do mesmo checkpoint, sem treinamento posterior. Houve reaproveitamento de 109.905 previsões com entradas idênticas e um complemento de 30.273 requisições. O complemento substituiu 182 previsões iSarcasm após restaurar os finais de linha canônicos; os registros históricos foram preservados localmente. Os seis arquivos de execução publicados são idênticos aos arquivos locais consolidados e pontuados pelo kit.

A reconstrução em Windows exigiu restaurar a ordem POSIX de API-Bank, metadados dos arquivos e o histórico completo de SGD. O SGD congelado inclui turnos futuros; o kit também documenta vazamento em RouterBench. Essas tarefas estão fora do índice. Os dados brutos, perguntas e gabaritos não são redistribuídos no dataset de previsões.

O resultado revela limitações amplas de generalização. No recorte textual MCQ do HLE, foram 74/501 acertos, abaixo da referência de acaso do kit. A triagem textual de exposição anterior não incluiu HLE nem as seis tarefas de exibição acrescentadas no complemento; a exposição de pré-treino e dos professores não é inteiramente conhecida. Os testes privados não foram executados e nenhuma posição oficial é reivindicada.
