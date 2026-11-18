# Previsão de casos de dengue em 10 capitais

Código do [artigo de João Ataíde](https://www.joaoataide.com/blog/previsao-de-series-temporais-com-o-prophet/).

Casos semanais de dengue (InfoDengue, Fiocruz e FGV) e clima de 10 capitais, de 2010 a 2026. Compara último valor, mesma semana do ano anterior, ARIMA, ETS, Prophet, ARIMA com clima defasado e LightGBM, em previsões de 4 semanas, com 40 origens por capital (uma a cada 8 semanas).

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `baixar_dados.py` | Baixa os dados do InfoDengue (API aberta) para `dados/dengue_semanal.csv` |
| `dengue.py` | Os sete primeiros modelos, as origens móveis e o resumo por capital. Grava `resultados/` |
| `dengue_global.py` | O LightGBM único para as 10 capitais, sem e com o Rt. Acrescenta as linhas ao `erros_origens.csv`; depois rode `python dengue.py --so-resumo` |

## Como rodar

```bash
pip install -r requirements.txt
python baixar_dados.py
python dengue.py --capitais "Recife,Rio de Janeiro" --saida resultados   # algumas capitais
python dengue.py                                                          # as 10, em várias horas
```

Para rodar em paralelo, execute `dengue.py` com `--capitais` e `--saida` diferentes e depois junte os `erros_origens.csv` em uma pasta e rode `python dengue.py --so-resumo`.

## Limites

- Casos são notificações, e a estimativa corrige o atraso, mas não o subregistro.
- Há poucos anos epidêmicos por capital, e o Prophet foi usado sem ajuste.
