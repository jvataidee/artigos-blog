# Desenhista de Figura Rica

Skill e resultados do artigo [Desenhista de Figura Rica: um agente de IA para desenhar e entender problemas complexos](https://www.joaoataide.com/blog/desenhista-de-figura-rica-agente-de-ia-para-entender-problemas-complexos/).

## A skill

[`figura-rica.skill`](figura-rica.skill) é um arquivo zip com a pasta `figura-rica/`: o orquestrador (`SKILL.md`), os seis papéis em `agents/` (analisador de transcrição, facilitador, leitor de conhecimento, desenhista, prompt visual e exportador para slides) e, em `scripts/`, o renderizador do estilo **Cena de Proveniência** (`render.html`, `desenhar.py` e uma especificação de exemplo).

A Cena de Proveniência é o estilo das figuras do artigo: linha contínua para o que foi informado, linha tracejada com "?" para hipótese, zigue-zague vermelho para conflito e um painel com a próxima pergunta e as lacunas. Essa convenção e o renderizador foram criados durante a execução dos três casos e incorporados à skill depois dela.

- **Claude.ai:** envie o arquivo na área de Skills das configurações.
- **Claude Code:** descompacte e coloque a pasta `figura-rica/` em `~/.claude/skills/` (para todos os projetos) ou em `.claude/skills/` (só no projeto).

Depois, peça algo como "monta uma Figura Rica desta situação" e descreva o problema. A skill não gera imagem por IA: ela estrutura o problema, desenha em código (SVG) ou entrega um prompt para uma ferramenta externa de imagem.

## Resultados do Claude nos três casos

Executados em 06/11/2026 no Claude Code com Claude Opus 5.5, sobre os mesmos enunciados do teste com ChatGPT e Gemini. Os papéis de facilitador e desenhista foram executados na própria conversa, sem subagentes separados.

| Arquivo | O que é |
|---|---|
| `resultados-claude/casoN_facilitador.md` | Saída do papel facilitador: fatos, hipóteses, lacunas, próxima pergunta mínima e estrutura da Figura Rica |
| `resultados-claude/casoN.json` | Especificação do desenho (atores, balões, ligações, painel de validação) |
| `resultados-claude/casoN.png` | Figura desenhada em código |
| `resultados-claude/render.html`, `desenhar.py` | Renderizador (rough.js + Playwright) que transforma a especificação em PNG |

Para redesenhar: `pip install playwright` e `python desenhar.py caso1.json` (usa o Google Chrome instalado e baixa o rough.js e a fonte Patrick Hand pela rede).

A capa do artigo foi desenhada com `capa_ebook.py`, no estilo das figuras do ebook (usa o `gifkit.py` do repositório do site, em `docs/ebook/ferramentas/gifs/`).
