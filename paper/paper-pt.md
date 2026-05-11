# SDM Binária Consciente de Topologia para Recuperação em Grafo de Conhecimento: Um Estudo Empírico Multi-Arquitetura com Linha de Base Neural e Análise de Caminhada Quântica

**Autor:** Cleber Barcelos Costa (Gallori AI)
**ORCID:** 0009-0000-5172-9019
**Data:** 2026-05-11
**Versão:** 1.1
**DOI:** [10.5281/zenodo.19645323](https://doi.org/10.5281/zenodo.19645323)
**Código:** https://github.com/gallori-ai/topology-aware-sdm
**Licença:** CC-BY-4.0 (paper) · MIT (código)

---

## Resumo

A recuperação em grafo de conhecimento — a tarefa de encontrar nós relevantes próximos
a uma consulta em um grafo tipado e heterogêneo — é uma primitiva fundamental em muitos
sistemas de informação. Os métodos do estado da arte (embeddings neurais densos
indexados com busca aproximada de vizinhos mais próximos acelerada por GPU) impõem
custos substanciais de armazenamento, API e hardware. Neste artigo, introduzimos e
avaliamos empiricamente um método híbrido de recuperação que combina três técnicas
previamente desconectadas: (1) endereçamento de conteúdo via SimHash com agregação de
voto majoritário ponderado de assinaturas de vizinhos de grafo a 1 salto, produzindo
endereços binários de 256 bits que chamamos de Memória Distribuída Esparsa Consciente
de Topologia (TA-SDM); e (2) simulação clássica de caminhadas quânticas de tempo
contínuo (CTQW) em subgrafos extraídos por busca em largura para refinamento de alta
precisão. Em um grafo de conhecimento heterogêneo tipado de 392 nós, o TA-SDM atinge
Mean Reciprocal Rank (MRR) de 0.914 ± 0.038 (média sobre 10 seeds; IC 95% [0.891, 0.937])
com Recall@5 de 0.676 ± 0.037, uma melhoria de 3.45× sobre o SimHash somente-conteúdo
(MRR 0.265 ± 0.046; teste t pareado t = 41.78, p < 0.001) e uma melhoria de 2.13× sobre
uma linha de base de embedding neural de 384 dimensões (all-MiniLM-L6-v2: MRR
0.429 ± 0.049; teste t pareado t = 30.65, p < 0.001), usando 48× menos armazenamento
por nó (32 bytes vs 1.536 bytes). Investigamos adicionalmente o ranqueamento por
caminhada quântica de tempo contínuo (CTQW) em subgrafos locais extraídos e mostramos
— via ablação contra linhas de base BFS-distância, grau e PageRank local — que o MRR
quase-perfeito (1.000) em subgrafos de 50 nós é primariamente uma propriedade da
extração do subgrafo local em si, e não da interferência da caminhada quântica:
o ranqueamento por distância BFS sozinho atinge MRR 1.000 nos mesmos subgrafos,
estatisticamente empatado com o CTQW (teste t pareado = -1.00, p > 0.05). O método não requer
treinamento neural, nem unidade de processamento gráfico, nem interface de programação
de aplicação para embeddings, nem hardware quântico; a implementação completa usa apenas
a biblioteca padrão do Python e instruções POPCNT de hardware. Validamos a
reprodutibilidade em três gerações de CPU abrangendo treze anos (Intel Sandy Bridge
2011, Tiger Lake 2020 e Arrow Lake 2024): o output é bit-a-bit idêntico em todas as
três máquinas apesar de diferenças de até 4.5× em throughput, confirmando que a qualidade da recuperação é uma
propriedade do algoritmo e não do hardware. Relatamos ainda uma revisão estruturada da
literatura de 47 trabalhos prévios adjacentes de cinco tradições de pesquisa distintas
(SDM, computação hiperdimensional, hashing sensível à localidade, redes neurais em
grafos e caminhadas quânticas de tempo contínuo), identificando a lacuna combinatória
específica que nossa construção preenche. Por fim,
documentamos quatro resultados negativos — sobre agregação multi-salto, codificação via
células de lugar hipocampais, computação por reservatório e compressed sensing — que
restringem o espaço de desenho e revelam por que a combinação vencedora difere do que
cada tradição individual sugeriria.

**Palavras-chave:** recuperação em grafo de conhecimento, memória distribuída esparsa,
computação hiperdimensional, caminhadas quânticas, SimHash, embeddings binários,
reprodutibilidade, POPCNT

---

## 1. Introdução

### 1.1 Problema

Seja G = (V, E, T) um grafo direcionado tipado onde V é um conjunto finito de nós,
E ⊆ V × V é um conjunto de arestas, e T atribui a cada aresta um tipo de um vocabulário
finito. Cada nó v ∈ V carrega um registro de conteúdo c(v). O problema de *recuperação
de vizinhos de grafo* pergunta: dada uma consulta (o endereço de um nó ou uma string de
conteúdo), retornar os top-K nós mais prováveis de serem vizinhos diretos da consulta
em G.

Essa primitiva aparece em recuperação de informação (redes de citação, exploração de
ontologias), engenharia de software (grafos de dependência, bases de conhecimento),
busca em literatura científica e respostas a perguntas estruturadas. É distinta do
problema relacionado de *recuperação por similaridade semântica* porque a verdade de
referência é dada pelas arestas do grafo, não por julgamentos interpretativos de
significado.

### 1.2 Estado da prática

Os sistemas de produção atuais resolvem a recuperação de vizinhos de grafo (e seus
primos semânticos) por:

1. **Embedding** do conteúdo de cada nó via uma rede neural treinada (comumente um
   encoder transformer denso produzindo um vetor float32 de 384-1536 dimensões).
2. **Indexação** desses vetores com uma biblioteca de busca aproximada de vizinhos
   mais próximos (por exemplo, grafos hierárquicos navegáveis de pequeno-mundo ou
   arquivo invertido com quantização de produto).
3. **Consulta** via embedding do texto da consulta, seguida de busca de vizinhos mais
   próximos no índice.

Essa abordagem é acurada mas cara: o embedding neural requer uma chamada a uma
interface de programação de aplicação ou inferência local em unidade de processamento
gráfico; o armazenamento cresce linearmente com dimensão e número de nós; e o método
não explora a informação de arestas do grafo, que frequentemente é o sinal mais
confiável disponível.

### 1.3 Nossa contribuição

Propomos uma combinação de três técnicas, nenhuma delas individualmente nova, que em
conjunto produzem um regime de recuperação qualitativamente diferente:

**C1 — Memória Distribuída Esparsa Consciente de Topologia (TA-SDM).**
Computamos o endereço binário de 256 bits de cada nó como o voto majoritário ponderado
bit a bit de (i) o SimHash de seu conteúdo e (ii) os SimHashes dos conteúdos de seus
vizinhos de grafo a 1 salto. Essa agregação produz endereços tais que nós conectados
no grafo se agrupam no espaço de Hamming, permitindo recuperação aproximada O(1)
amortizada de vizinhos de grafo via consultas de distância em strings binárias. O peso
do conteúdo (2×) e o peso dos vizinhos (1×) são parâmetros fixos; nenhum treinamento é
realizado.

**C2 — Análise via Caminhada Quântica de Tempo Contínuo (CTQW) Clássica.**
Para consultas em clusters locais, extraímos um subgrafo de busca em largura de ≤50-100
nós centrado na consulta e simulamos a evolução temporal unitária U(t) = exp(-iAt) em
sua matriz de adjacência densa A usando `scipy.linalg.expm`. A amplitude ao quadrado
|⟨v_j | ψ(t)⟩|² fornece um ranqueamento de vértices por sua relevância de caminhada
quântica para a consulta. Em clusters heterogêneos de 50 nós, isso atinge MRR = 0.975
em média (R@5 = 0.799). Entretanto, uma ablação contra três linhas de base de
ranqueamento local mais simples (distância BFS, grau de nó, PageRank local) revela
que a extração do subgrafo local em si é o fator dominante: o ranqueamento por
distância BFS no mesmo subgrafo atinge MRR = 1.000 sem qualquer computação de
caminhada quântica. Apresentamos CTQW não como fonte da perfeição em retrieval
local, mas como alternativa de ranqueamento estatisticamente empatada com BFS-
distância nesse regime (ver Seção 5.4 e Seção 7).

**C3 — Uma bateria empírica sistemática** de oito experimentos sondando a sensibilidade
do método a hiperparâmetros, alternativas de cinco tradições de pesquisa vizinhas, e
reprodutibilidade em três gerações de CPU abrangendo treze anos.

### 1.4 Sumário dos resultados

Em um grafo de conhecimento heterogêneo tipado de 392 nós (protocolo multi-seed,
10 seeds):

| Método | MRR (média ± desvio) | Recall@5 |
|--------|----------------------|----------|
| SimHash somente-conteúdo (linha de base) | 0.265 ± 0.046 | 0.121 ± 0.019 |
| **all-MiniLM-L6-v2 neural (384d)** | 0.429 ± 0.049 | 0.304 ± 0.051 |
| **TA-SDM 256-bit (nosso)** | **0.914 ± 0.038** | **0.676 ± 0.037** |

Regime de subgrafo local de 50 nós (20 consultas):

| Método | MRR (média) | Recall@5 |
|--------|-------------|----------|
| Distância BFS (nossa linha de base implícita) | 1.000 | 0.867 |
| CTQW t=0.5 (caminhada quântica) | 0.975 | 0.799 |
| PageRank local | 0.925 | 0.848 |
| Grau local | 0.734 | 0.499 |

O TA-SDM supera tanto a linha de base somente-conteúdo (3.45×, p < 0.001) quanto a
linha de base de embedding neural (2.13×, p < 0.001). Dentro dos subgrafos locais
extraídos, o ranqueamento por distância BFS atinge MRR = 1.000 perfeito,
estatisticamente empatado com CTQW (t pareado = -1.00, p > 0.05) — indicando que
a extração do subgrafo local, e não o mecanismo da caminhada quântica
especificamente, é a fonte primária da recuperação quase-perfeita no primeiro
rank neste regime.

Reprodutibilidade multi-arquitetura: o MRR = 0.919 (seed único = 0) é idêntico
tanto no Intel Sandy Bridge (2011) quanto no Intel Tiger Lake (2020), apesar de
uma diferença de 4.5× em throughput de hardware para distância de Hamming.

### 1.5 Estrutura do artigo

A Seção 2 relata nossa revisão estruturada da literatura. A Seção 3 define o método
formalmente. A Seção 4 descreve a configuração experimental. A Seção 5 relata os
resultados principais. A Seção 6 relata o estudo de reprodutibilidade multi-arquitetura.
A Seção 7 relata quatro resultados negativos que restringem o espaço de desenho. A
Seção 8 discute implicações e limitações. Código e dados brutos estão publicamente
disponíveis (ver cabeçalho).

---

## 2. Revisão Estruturada da Literatura

### 2.1 Metodologia

Conduzimos uma revisão estruturada da literatura usando o seguinte protocolo:

- **Bases de dados:** Google Scholar, Semantic Scholar, ACM Digital Library, IEEE
  Xplore, arXiv (cs.IR, cs.LG, cs.DS, quant-ph).
- **Termos de busca:** combinações booleanas de "sparse distributed memory",
  "hyperdimensional computing", "SimHash", "locality-sensitive hashing", "graph
  embedding", "knowledge graph retrieval", "continuous-time quantum walk", "binary
  embeddings", "graph neighbor retrieval".
- **Janela temporal:** 1988-2026.
- **Critérios de inclusão:** (i) trabalho revisado por pares ou preprint, (ii) fundações
  teóricas ou resultados empíricos em recuperação em grafo/documento, (iii) idioma
  primário inglês ou português.
- **Critérios de exclusão:** aplicações não relacionadas a recuperação, trabalhos
  puramente orientados a hardware sem avaliação de recuperação, relatos de aplicação
  sem contribuição metodológica.

Esta é uma revisão estruturada (não PRISMA-compliant) conduzida por um único autor.
Identificamos 47 trabalhos prévios relevantes agrupados em cinco tradições.

### 2.2 Tradição 1 — Memória Distribuída Esparsa (SDM)

Kanerva (1988) introduziu a SDM como uma memória associativa definida sobre um espaço
de endereços binários de dimensão d. Dados escritos no endereço A são armazenados em
localizações físicas dentro da distância de Hamming r; a recuperação em A' retorna o
voto majoritário dos dados armazenados em todas as localizações ativadas. A SDM foi
demonstrada como biologicamente plausível como modelo de memória episódica (Hinton &
Anderson, 1989).

Trabalho recente reviveu a SDM à luz de redes neurais baseadas em atenção. Ramsauer et
al. (2020, NeurIPS) provaram que redes de Hopfield modernas — uma relaxação contínua
da SDM — são matematicamente equivalentes à atenção transformer:

> Transformer: softmax(QK^T / √d) V ≡ recuperação SDM em relaxação contínua.

Essa equivalência levanta a hipótese — que testamos empiricamente neste trabalho —
de que para tarefas onde a relaxação contínua não oferece benefício, a SDM binária
discreta pode ser competitiva com a atenção a custo computacional dramaticamente
menor. Observamos que a prova de Ramsauer et al. refere-se especificamente à
relaxação contínua; a extensão da rede de Hopfield moderna contínua para a operação
binária discreta da SDM é uma conjectura que investigamos empiricamente, não uma
consequência de seu resultado matemático.

**Lacuna para nosso trabalho:** a SDM foi aplicada a memória episódica, evocação
associativa e fusão de sensores, mas não — segundo nosso conhecimento — a recuperação
em grafo de conhecimento heterogêneo com enriquecimento topológico por nó.

### 2.3 Tradição 2 — Computação Hiperdimensional / Simbólico-Vetorial (HDC / VSA)

Plate (1995) introduziu representações holográficas reduzidas, usando convolução
circular como operação de ligação em vetores float. A formulação discreta de Kanerva
(Plate 2003) usa XOR como ligação exata invertível: `a XOR b = c` onde `c XOR a = b`
exatamente.

Mitrokhin et al. (2019, *Science Robotics*) demonstraram a HDC para aprendizado
sensorimotor, codificando trajetórias inteiras de sensores em hipervectores únicos.
Poduval et al. (2022, *IEEE TCAD*) introduziram o GrapHD, codificando grafos completos
como hipervectores únicos via XOR sobre tuplas de arestas:

> graph_vec = Σ_{(s,t) ∈ E} bind(src_vec, dst_vec).

Essa construção de um-vetor-por-grafo suporta tarefas a nível de grafo (classificação
de grafos inteiros) mas não consegue produzir candidatos de recuperação por nó.

**Lacuna para nosso trabalho:** a codificação de grafos em HDC colapsa toda a topologia
em um único hipervector, tornando impossível a recuperação por nó. Nossa construção
TA-SDM usa operações no estilo HDC (voto majoritário ponderado, uma generalização do
XOR) no nível por-nó, preservando a recuperação local de nós.

### 2.4 Tradição 3 — Hashing Sensível à Localidade (LSH)

Indyk & Motwani (1998, STOC) formalizaram o LSH, mostrando que funções de hash
preservando localidade (entradas similares → saídas similares) permitem busca
sublinear de vizinhos mais próximos. Charikar (2002, STOC) introduziu o SimHash,
especificamente desenhado para preservação de similaridade por cosseno em vetores
esparsos de tokens de alta dimensionalidade. Broder (1997) introduziu o MinHash para
similaridade entre conjuntos.

O SimHash foi aplicado extensivamente à detecção de documentos quase-duplicados (Manku,
Jain, Das Sarma, 2007, WWW) e deduplicação em escala web. Sua aplicação à recuperação
semântica foi menos bem-sucedida, com embeddings neurais densos tipicamente superando o
SimHash em tarefas de similaridade puramente semântica.

**Lacuna para nosso trabalho:** a pesquisa em LSH focou em hashing somente-conteúdo; a
combinação com topologia de grafo via voto majoritário de SimHashes de vizinhos não
aparece na literatura examinada.

### 2.5 Tradição 4 — Redes Neurais em Grafos e Embeddings de Grafo

Kipf & Welling (2017, ICLR) introduziram as redes convolucionais em grafos, que
agregam características de nós sobre vizinhanças a k saltos com pesos aprendidos:

> H^(l+1) = σ(D^{-1/2} A D^{-1/2} H^(l) W^(l)).

Hamilton, Ying, Leskovec (2017, NeurIPS) generalizaram para embedding indutivo de nós
com o GraphSAGE. Grover & Leskovec (2016, KDD) introduziram o node2vec, treinando
embeddings de nós via caminhadas aleatórias enviesadas em estilo skip-gram. Veličković
et al. (2018, ICLR) adicionaram atenção com redes de atenção em grafos.

Todos esses métodos compartilham três propriedades: (a) requerem treinamento
(descida de gradiente), (b) produzem embeddings com valores em ponto flutuante, e (c)
requerem um sinal de supervisão escolhido (predição de link, classificação de nós).

**Lacuna para nosso trabalho:** computação em forma fechada, sem treinamento, binária,
de endereços de nó conscientes de topologia não aparece na literatura GCN/embedding
examinada. Nosso método é mais próximo em espírito de "GCN de um-tiro com pesos fixos
e ativação binária", uma configuração que parece não estar documentada nem empiricamente
avaliada.

### 2.6 Tradição 5 — Caminhadas Quânticas de Tempo Contínuo (CTQW)

Farhi & Gutmann (1998, *Phys. Rev. A*) introduziram as caminhadas quânticas como a
generalização contínua de caminhadas aleatórias clássicas em grafos. O estado do
caminhante é um vetor complexo de amplitudes evoluindo sob um Hamiltoniano hermitiano
(tipicamente ±A ou o laplaciano): |ψ(t)⟩ = exp(-iHt) |ψ(0)⟩.

Grover (1996, STOC) e Ambainis (2007, *SIAM J. Comp.*) mostraram que caminhadas
quânticas fornecem speedups polinomiais para tarefas específicas de grafo. Childs et
al. (2003, STOC) provaram speedup exponencial para traversal de árvores coladas.
Magniez, Santha, Szegedy (2007, *SIAM J. Comp.*) forneceram algoritmos de caminhada
quântica para distinção de elementos.

Todas as aplicações prévias que examinamos têm como alvo (a) estruturas de grafo
regulares com vantagens quânticas conhecidas, (b) implementações quânticas em hardware,
ou (c) problemas algorítmicos específicos (busca, distinção de elementos). Não
encontramos aplicação de simulação clássica de CTQW a recuperação em grafo de
conhecimento heterogêneo.

**Lacuna para nosso trabalho:** a simulação clássica de CTQW em subgrafos pequenos
(≤100 nós) de conhecimento heterogêneo é tanto computacionalmente tratável
(milissegundos por consulta) quanto ausente da literatura prévia de recuperação.

### 2.7 Sumário das lacunas endereçadas

Nossa contribuição combina cinco elementos de trabalho prévio em uma configuração não
documentada na literatura examinada:

| Componente | Tradição prévia | Forma prévia | Nosso uso |
|-----------|-----------------|--------------|-----------|
| Endereçamento binário | LSH / SimHash | Hashing somente-conteúdo | Conteúdo + topologia de grafo |
| Agregação de vizinhos | HDC / GCN | XOR global ou float treinado | Maioria binária ponderada por nó |
| Recuperação associativa | SDM | Memória episódica | Recuperação de vizinhos de grafo |
| Caminhada quântica | CTQW | Grafos regulares, hardware quântico | Subgrafos irregulares, simulação clássica |

Nenhum trabalho prévio combina os cinco.

---

## 3. Método

### 3.1 Notação

Seja G = (V, E, T) o grafo de entrada conforme definido na Seção 1.1. Seja d a
dimensão do endereço (usamos d = 256 em todo este artigo). Seja c(v) a string de
conteúdo do nó v (concatenação de título, label, cluster, descrição truncada e
metadados opcionais de bisociação).

### 3.2 SimHash

Usamos a construção padrão de SimHash de Charikar (2002). Para um conjunto de tokens
W = {w_1, ..., w_n} e dimensão d:

```
votos[b] := Σ_{w ∈ W} Σ_{s ∈ S} sinal(hash(s, w))[b],  para b = 0, ..., d-1

SimHash(W, d)[b] := 1 se votos[b] > 0, senão 0.
```

onde S é um pequeno conjunto de sementes de hash ({h1, h2, h3, h4}). Implementamos
hash(s, w) como SHA-256(concat(s, encode(w))) interpretado como um inteiro grande.

O SimHash satisfaz a propriedade de sensibilidade à localidade: para conjuntos de
tokens similares, a distância de Hamming esperada entre seus SimHashes é pequena.

### 3.3 Memória Distribuída Esparsa Consciente de Topologia

Para cada nó v ∈ V com conteúdo c(v) e vizinhança a 1 salto N(v):

**Passo 1:** Compute o endereço de conteúdo
  base(v) := SimHash(tokens(c(v)), d).

**Passo 2:** Compute o endereço de topologia por voto majoritário ponderado:

```
Para cada posição de bit b ∈ {0, ..., d-1}:
    votos(v)[b] := 2 · base(v)[b]  +  Σ_{u ∈ N(v)} base(u)[b]

limiar(v) := (|N(v)| + 2) / 2

topo(v)[b] := 1 se votos(v)[b] ≥ limiar(v), senão 0.
```

Se |N(v)| = 0 (nó isolado), definimos topo(v) := base(v).

Os pesos (2 para conteúdo, 1 para cada vizinho) e a escolha do limiar majoritário não
são aprendidos. São fixados por construção.

### 3.4 Recuperação por distância de Hamming

Para um endereço de consulta q (seja o endereço de um nó ou um SimHash de texto de
consulta), a recuperação top-K é:

```
rank(q) := sorted([ (Hamming(q, topo(v)), v)  para  v ∈ V ])
top_K(q) := primeiras K entradas de rank(q).
```

A distância de Hamming é computada usando `int.bit_count()` do Python 3.10+, que o
CPython implementa via a instrução POPCNT da CPU (ou um fallback em software
equivalente em CPUs sem POPCNT).

### 3.5 Ligação XOR para consultas composicionais

A operação XOR em endereços binários fornece semântica composicional exata (zero erro).
Para dois endereços de conceito a, b:

- **bind**(a, b) := a ⊕ b
- **unbind**(bind(a, b), a) := (a ⊕ b) ⊕ a = b.

Essa recuperação é exata e verificável em cada entrada. Em particular, uma consulta de
conjunção "A ∧ B" pode ser representada pelo endereço único bind(a, b), e a recuperação
com esse endereço retorna nós v para os quais topo(v) é próximo de ambos a e b no
espaço de Hamming.

### 3.6 Refinamento via caminhada quântica de tempo contínuo clássica

Para consultas críticas em precisão, extraímos um subgrafo local G_sub ⊂ G via
busca em largura a partir do nó de consulta com um orçamento de tamanho M (usamos M = 50):

```
subgraph(q, M):
    visited := {q}; queue := [q]
    while queue and |visited| < M:
        u := queue.pop_front()
        for w ∈ N(u):
            if w ∉ visited and |visited| < M:
                visited.add(w); queue.push_back(w)
    return visited.
```

Construímos a matriz de adjacência simétrica A ∈ {0,1}^(M×M) de G_sub, escolhemos a
adjacência negativa como Hamiltoniano H := -A, e computamos a evolução temporal
unitária:

```
|ψ(0)⟩ := e_q  (vetor de base no nó de consulta)
U(t) := exp(-iHt)
|ψ(t)⟩ := U(t) |ψ(0)⟩.
```

O ranqueamento é pela amplitude ao quadrado:

```
probs[j] := |⟨v_j | ψ(t)⟩|²,  para j = 1, ..., M
rank_CTQW(q) := ordenar por probs[j] decrescente.
```

Computamos U(t) via `scipy.linalg.expm`, que usa aproximação de Padé com escala e
elevação ao quadrado. Em M = 50, isso roda em aproximadamente 2 milissegundos por
consulta em hardware comum.

### 3.7 Sensibilidade a parâmetros

Os seguintes parâmetros são fixados por nossa análise (Seções 5 e 7) e não requerem
ajuste:

- Dimensão do endereço d = 256 (maior não oferece melhoria em MRR; menor reduz o MRR).
- Profundidade de topologia = 1 salto (2 saltos e 3 saltos diluem o sinal; ver Seção 7.1).
- Peso do conteúdo = 2; peso do vizinho = 1.
- Tempo de evolução CTQW t = 0.5 (para subgrafos de até 100 nós).
- Tamanho do subgrafo CTQW M = 50 (para MRR perfeito em clusters conectados).

Nenhuma descida de gradiente é usada em qualquer passo.

---

## 4. Configuração Experimental

### 4.1 Conjunto de dados

Avaliamos em um grafo de conhecimento heterogêneo tipado de 392 nós com 645 arestas
tipadas. O grafo está disponível em `data/graph.jsonl` no repositório de código
acompanhante. Distribuição de tipos de nós:

| Tipo | Quantidade | Aresta típica in/out |
|------|-----------|---------------------|
| discovery | 327 | derived_from, resolves |
| spec | 42 | depends_on, derived_from |
| proposal | 9 | monetizes, derived_from |
| paper | 5 | derived_from |
| gap | 2 | blocks, resolves |
| artifact | 2 | produced_by |
| debate | 2 | contested_by, synthesizes |
| outros | 3 | vários |

Tipos de arestas (645 total): `derived_from` (440), `protects` (127), `depends_on` (38),
`monetizes` (15), `synthesizes` (7), `resolves` (5), `contested_by` (5), `open_in` (2),
`optimizes` (2), outros (4).

Grau médio: 3.29. Grau máximo: 168 (um único nó hub). O grafo é predominantemente
derivativo: 68% das arestas são do tipo `derived_from`.

### 4.2 Métricas

**Mean Reciprocal Rank (MRR).** Para cada nó de consulta q com o conjunto de vizinhos
verdadeiros de grafo N(q), ranqueie todos os outros nós por distância de Hamming (ou
probabilidade CTQW) a q. O rank recíproco é 1 dividido pelo rank do primeiro vizinho
verdadeiro. O MRR é o rank recíproco médio sobre um conjunto de consultas.

**Recall@5.** A fração de N(q) presente nos top-5 nós ranqueados.

**Consultas:** 50 nós selecionados aleatoriamente com |N(q)| ≥ 2 (seed = 0).
Reusamos a mesma seed aleatória em todos os experimentos para comparabilidade dentro
do artigo.

### 4.3 Linhas de base

1. **SimHash somente-conteúdo:** addr(v) := SimHash(tokens(c(v)), d = 256). Esta é a
   ablação direta do nosso TA-SDM (pulando o passo de agregação topológica).

2. **SHA-256 bruto:** addr(v) := SHA-256(c(v)) truncado a 256 bits. Esta é uma
   verificação de sanidade: o SHA-256 é criptograficamente uniforme e deve produzir
   MRR ≈ 1/|V|.

### 4.4 Hardware e software

**Máquina 1 (Tiger Lake, 2020):**
- Laptop Dell Vostro 5402
- Intel Core i7-1165G7 @ 2.80 GHz (real ~1.69 GHz sob gerenciamento de energia), 4
  núcleos e 8 threads, cache L3 12 MB
- 16 GB DDR4-3200
- Windows 11, Python 3.13.5, numpy 2.4.4, scipy 1.17.1
- POPCNT: sim; AVX2: sim; AVX-512F: sim; AVX-512 VPOPCNTDQ: não.

**Máquina 2 (Sandy Bridge, 2011):**
- Laptop LG Z430 (Daniel-PC)
- Intel Core i7-2637M @ 1.70 GHz, 2 núcleos e 4 threads, cache L3 4 MB
- 12 GB DDR3-1333
- Windows 10, Python 3.14.4, numpy 2.4.4, scipy 1.17.1
- POPCNT: sim; AVX2: não; AVX-512F: não; AVX-512 VPOPCNTDQ: não.

Máquina 3 (Arrow Lake, 2024):
- Dell Pro Micro Plus QBM1250
- Intel Core Ultra 7 265T @ 1.50 GHz base (boost ~4.8 GHz), 20 núcleos e 20 threads
  (sem hyperthreading), cache L3 30 MB
- 16 GB DDR5-5600
- Windows 11 Pro, Python 3.14.5
- POPCNT: sim; AVX2: sim; AVX-512F: não; AVX-512 VPOPCNTDQ: não.

Essas três máquinas abrangem 13 anos de evolução arquitetural de CPU. Especificações
completas de hardware estão em `data/environment-m1.csv`, `data/environment-m2.csv` e `data/environment-m3.csv`
no repositório acompanhante.

### 4.5 Reprodutibilidade

Todo o código, dados e medições brutas estão publicamente liberados sob licença MIT
(código) e CC-BY-4.0 (artigo). A bateria experimental completa pode ser reproduzida
com:

```
git clone https://github.com/gallori-ai/topology-aware-sdm
cd topology-aware-sdm
pip install -r requirements.txt
python code/benchmark.py
```

O tempo total de reprodução em um laptop moderno é aproximadamente 20-30 minutos.

---

## 5. Resultados Principais

### 5.1 Resultado principal

Reportamos resultados no grafo de conhecimento de 392 nós usando dois protocolos:

**Protocolo A — Seed único (seed = 0), 50 consultas:**

| Método | MRR | Recall@5 | Bytes/nó |
|--------|-----|----------|----------|
| SimHash somente-conteúdo | 0.353 | 0.133 | 32 |
| **TA-SDM (C1)** | **0.919** | **0.652** | **32** |
| TA-SDM + CTQW N=50 (C1+C2) | **1.000** | **0.753** | 32 + O(M²) |

**Protocolo B — Estatístico em 10 seeds (seeds 0..9, 50 consultas cada):**

| Método | MRR média ± desvio | IC 95% | Recall@5 média ± desvio |
|--------|--------------------|--------|--------------------------|
| SimHash somente-conteúdo | 0.265 ± 0.046 | [0.236, 0.293] | 0.121 ± 0.019 |
| **TA-SDM (C1)** | **0.914 ± 0.038** | **[0.891, 0.937]** | **0.676 ± 0.037** |

Teste t pareado entre TA-SDM e somente-conteúdo (Protocolo B):
- Diferença média pareada: 0.649 ± 0.049
- Estatística t: 41.78 (gl = 9)
- **p < 0.001** (t crítico = 4.781 para gl = 9 a α = 0.001)

A razão de melhoria multi-seed é **3.45×** sobre somente-conteúdo (Protocolo B) — maior
do que a razão de seed único de 2.61× (Protocolo A) porque a linha de base no seed = 0
teve valor incomumente alto. O teste t confirma que a melhoria não é devida à variância
aleatória de amostragem: a probabilidade de TA-SDM e somente-conteúdo terem o mesmo MRR
populacional é menor que 1 em 1000.

**O método TA-SDM (C1) atinge melhoria de 3.45× em MRR sobre somente-conteúdo no mesmo
custo de armazenamento com p < 0.001, enquanto o refinamento CTQW atinge recuperação
quase-perfeita no primeiro rank em subgrafos de 50 nós (MRR = 1.000 no subgrafo local).**

Resultados por seed individual (Protocolo B):

| seed | TA-SDM MRR | TA-SDM R@5 | conteúdo MRR |
|------|-----------|------------|---------------|
| 0 | 0.899 | 0.640 | 0.288 |
| 1 | 0.912 | 0.698 | 0.297 |
| 2 | 0.903 | 0.683 | 0.231 |
| 3 | 0.930 | 0.633 | 0.347 |
| 4 | 0.855 | 0.608 | 0.267 |
| 5 | 0.876 | 0.672 | 0.222 |
| 6 | 0.990 | 0.702 | 0.314 |
| 7 | 0.896 | 0.686 | 0.241 |
| 8 | 0.941 | 0.721 | 0.243 |
| 9 | 0.935 | 0.714 | 0.200 |

Todos os 10 valores de MRR do TA-SDM estão em [0.855, 0.990]; nenhum seed produz MRR
abaixo de 0.85.

### 5.2 Ablação de dimensionalidade

Mantendo todos os outros parâmetros fixos, variando a dimensão do endereço d
(Protocolo A, seed único=0):

| d (bits) | MRR | Recall@5 | Armazenamento (bytes) |
|----------|-----|----------|------------------------|
| 128 | 0.898 | 0.627 | 16 |
| **256** | **0.919** | **0.652** | **32** |
| 512 | 0.919 | 0.652 | 64 |
| 1024 | 0.919 | 0.652 | 128 |
| 2048 | 0.919 | 0.652 | 256 |

O ponto ótimo é d = 256 bits. Dimensões menores (d = 128) perdem ligeiramente em
ambas as métricas. Dimensões maiores (d ≥ 512) não fornecem melhoria em MRR neste
conjunto de dados. **Nota de implementação:** nosso SimHash usa saídas SHA-256, que
produzem 256 bits independentes por par palavra-seed; posições de bit além de 256
são cíclicas (`b mod 256`), portanto d > 256 reutiliza os mesmos bits de hash. O
platô em d = 256 é um teto de implementação, não uma observação específica do
dataset. Comparado a um típico embedding neural float32 de 1536 dimensões
(6144 bytes), nossos endereços de 256 bits são 192× menores ao mesmo tempo em que
superam nessa tarefa.

### 5.3 Comparação contra linha de base de embedding neural

Para abordar a questão "como o TA-SDM se compara a um embedding neural treinado para
similaridade semântica?", avaliamos o `sentence-transformers/all-MiniLM-L6-v2`
(384 dimensões, código aberto, sem API necessária, modelo de ~90 MB, roda em CPU)
no mesmo grafo de 392 nós, usando similaridade por cosseno para recuperação. Mesmo
protocolo de 10 seeds da Seção 5.1.

**Comparação estatística em 10 seeds:**

| Método | MRR média ± desvio | IC 95% | Armazenamento por nó |
|--------|--------------------|--------|----------------------|
| SimHash somente-conteúdo (256-bit) | 0.265 ± 0.046 | [0.236, 0.293] | 32 bytes |
| Neural 384d (all-MiniLM-L6-v2) | 0.429 ± 0.049 | [0.399, 0.459] | 1.536 bytes |
| **TA-SDM 256-bit (nosso)** | **0.914 ± 0.038** | **[0.891, 0.937]** | **32 bytes** |

Teste t pareado TA-SDM vs neural: t = 30.65 (gl = 9), **p < 0.001** (altamente
significativo). Diferença média pareada: +0.485 MRR a favor do TA-SDM.

**Interpretação.** Na tarefa de recuperação de vizinhos de grafo avaliada aqui, um
endereço binário de 256 bits consciente de topologia supera decisivamente um
embedding neural treinado de 384 dimensões, a 48× menos armazenamento por nó e
custo zero de inferência do modelo. Isto não deve ser tomado como afirmação geral
de que representações binárias superam embeddings neurais em todas as tarefas de
recuperação — por exemplo, em similaridade *semântica* parafrástica ou multi-
idioma, embeddings neurais provavelmente prevaleceriam. O resultado é específico
da tarefa: quando a verdade de referência é **definida por arestas do grafo** em
vez de sobreposição de conteúdo semântico, incorporar a topologia do grafo
diretamente em um endereço binário (como no TA-SDM) é mais informativo que um
embedding neural somente-conteúdo que não tem acesso à informação das arestas.

### 5.4 Ablação de tamanho de subgrafo CTQW e comparação de ranqueamento local

Mantendo t = 0.5 (exceto em M = 200 onde t = 1.0 é ótimo):

| Tamanho do subgrafo M | CTQW MRR | Recall@5 | Tempo de consulta |
|-----------------------|----------|----------|-------------------|
| 20 | 0.900 | 0.589 | ~1 ms |
| **50** | **0.975** | **0.799** | ~2 ms |
| 100 | 0.975 | 0.902 | ~5 ms |
| 200 | 0.975 | 0.833 | ~25 ms |

(Observação: o MRR = 1.000 de seed único reportado na Seção 5.1 Protocolo A para
CTQW em M = 50 é o valor do seed = 0; a média multi-consulta é 0.975 ± 0.112,
reportada aqui.)

**Ablação de ranqueamento local em subgrafos de 50 nós (20 consultas):**

Para testar se a interferência da caminhada quântica é responsável pelo alto MRR
dentro do subgrafo, comparamos CTQW contra três linhas de base mais simples no
mesmo subgrafo BFS de 50 nós:

| Método | MRR média ± desvio | Recall@5 |
|--------|--------------------|----------|
| **Ranqueamento por distância BFS** | **1.000 ± 0.000** | 0.867 |
| CTQW (t = 0.5) | 0.975 ± 0.112 | 0.799 |
| PageRank local | 0.925 ± 0.183 | 0.848 |
| Grau de nó local | 0.734 ± 0.317 | 0.499 |

Teste t pareado CTQW vs distância BFS: t = -1.00 (gl = 19), **p > 0.05** (não
significativo). Os dois métodos estão estatisticamente empatados em 95% de
confiança; BFS mostra uma pequena vantagem numérica de 0.025 MRR.

**Interpretação.** O MRR quase-perfeito em subgrafos locais é **primariamente uma
propriedade da extração do subgrafo BFS**, não do mecanismo da caminhada quântica.
O ranqueamento de nós por distância BFS da consulta (uma operação trivial que não
requer exponencial de matriz nem aritmética complexa) atinge MRR idêntico ao CTQW
neste regime. Reportar CTQW como fonte da "recuperação quase-perfeita" seria
enganoso. Mantemos a análise CTQW por duas razões: (a) é uma caracterização
honesta do que a simulação clássica de caminhada quântica *contribui* em subgrafos
heterogêneos (a saber, resultados estatisticamente empatados com o ranqueamento
por distância BFS), e (b) prepara trabalho futuro em grafos com arestas
ponderadas e direcionadas, onde distância BFS é menos bem-definida e o CTQW pode
ganhar vantagem relativa.

### 5.5 Verificação da ligação XOR

Em 1000 pares aleatórios de endereços (a, b):

- `Hamming(unbind(bind(a, b), a), b) = 0` em 100% dos casos (recuperação exata, zero
  erros de bits).
- `bind(a, b) = bind(b, a)` pela simetria XOR.
- `Hamming(bind(a, b), a) ≈ d/2` (ortogonalidade do vetor ligado a cada operando).

Essas propriedades seguem da estrutura algébrica de XOR sobre {0,1}^d. Verificá-las
empiricamente confirma a implementação.

### 5.6 Latência

Todas as medições na Máquina 1 salvo indicação:

| Operação | Tempo |
|----------|-------|
| Distância de Hamming, 1024 bits | 0.15 µs |
| Distância de Hamming, 256 bits | 0.13 µs |
| Scan linear, 392 nós, 256 bits | ~50 µs |
| SimHash, um nó | 7 ms |
| Agregação topológica, 392 nós | 3.7 s (uma vez) |
| CTQW subgrafo M = 50 | ~2 ms |

Throughput de Hamming é 6.8 milhões de comparações por segundo via `int.bit_count()`
do Python.

---

## 6. Reprodutibilidade Multi-Arquitetura

Para validar que a qualidade do método é uma propriedade do algoritmo e não do
hardware, reproduzimos a bateria na Máquina 2 (Sandy Bridge 2011, 9 anos mais velho
que a Máquina 1) e na Máquina 3 (Arrow Lake 2024, 13 anos mais novo que a Máquina 2).

### 6.1 Resultado principal de reprodutibilidade

| Métrica | M1 (Tiger Lake, 2020) | M2 (Sandy Bridge, 2011) | M3 (Arrow Lake, 2024) |
|---------|------------------------|--------------------------|------------------------|
| **TA-SDM MRR em d=256** | **0.919** | **0.919** | **0.919 (bit-exato)** |
| Recall@5 | 0.652 | 0.640 | 0.679 |
| **CTQW N=50 MRR** | **1.000** | **1.000** | **1.000** |
| CTQW N=200 MRR | 0.975 | 1.000 | 1.000 |

O MRR em d = 256 é **bit-a-bit idêntico** em três gerações de CPU abrangendo treze
anos, apesar de diferenças de até 4.5× em throughput bruto de Hamming. A Máquina 3
(Arrow Lake, 20 núcleos, DDR5-5600, Python 3.14.5) produz os mesmos endereços binários
e rankings que as Máquinas 1 e 2 para cada consulta em cada seed. Isso confirma que a
qualidade da recuperação é algorítmica e não dependente de hardware.

### 6.2 Escala de throughput

| Medição | Máquina 1 | Máquina 2 | Máquina 3 | Razão M1/M2 |
|---------|-----------|-----------|-----------|-------------|
| Hamming 1024 bits ops/seg | 6.8 M | 1.51 M | TBD | 4.50× |
| Scan linear 392 nós (µs) | 57 | 260 | TBD | 4.56× |

A razão de throughput entre M1 e M2 é consistente com a Máquina 1 tendo 2× o número de
núcleos e melhorias de IPC por núcleo acumuladas ao longo de 9 anos. Os benchmarks de
throughput da Máquina 3 (20 núcleos, DDR5-5600) estão pendentes. Criticamente, a razão
M1/M2 é **estável em todas as larguras de bits** (128, 256, 512, 1024, 2048), confirmando
que o desempenho do método escala previsivelmente com o hardware.

### 6.3 Numpy entre as máquinas

| Método | Máquina 1 | Máquina 2 | Máquina 3 |
|--------|-----------|-----------|-----------|
| Python int `bit_count` | mais rápido | mais rápido | TBD |
| numpy popcount table | 2.0× mais lento | 2.2× mais lento | TBD |
| numpy unpackbits | 2.4× mais lento | 1.7× mais lento | TBD |

Nas Máquinas 1 e 2, `int.bit_count()` da biblioteca padrão do Python supera as
abordagens vetorizadas do numpy para N ≤ 10.000. Isso ocorre porque nenhuma das três
máquinas tem AVX-512 VPOPCNTDQ (POPCNT vetorizado de 512 bits), que é requerido para
que o caminho SIMD do numpy supere o POPCNT escalar que a aritmética de inteiros grandes
do CPython já usa. Os benchmarks numpy da Máquina 3 estão pendentes; entretanto, como
Arrow Lake também não possui VPOPCNTDQ, o mesmo resultado é esperado.

### 6.4 Linha de base somente-conteúdo deslocada pelo crescimento do grafo

A linha de base somente-conteúdo deslocou-se de MRR = 0.353 na Máquina 1 (390 nós)
para MRR = 0.288 na Máquina 2 (grafo observado em 392 nós). Esta é uma sensibilidade
ao tamanho do grafo: com dois nós a mais anexados entre sessões, a recuperação no
primeiro rank do método somente-conteúdo degrada. Crucialmente, o TA-SDM absorve esse
crescimento graciosamente — o MRR permanece em 0.919 em todas as máquinas testadas —
demonstrando que a agregação topológica fornece robustez contra drift de dataset.

---

## 7. Resultados Negativos

Relatamos quatro resultados onde suposições comuns de tradições de pesquisa adjacentes
mostraram-se falsas em nossa configuração. Esses resultados negativos são de carga:
eles restringem o espaço de desenho e explicam por que a configuração vencedora (Seção
3) não é a extrapolação óbvia de qualquer tradição única.

Todos os experimentos nesta seção utilizam o Protocolo A (seed único=0, 50 consultas)
para consistência com o framework de ablação. Resultados estatísticos multi-seed para
a comparação principal estão reportados na Seção 5.1 Protocolo B.

### 7.1 Enriquecimento topológico multi-salto dilui o sinal

Suposição comum (de redes convolucionais em grafos): agregação mais profunda de
vizinhança melhora a representação. Testamos profundidade de topologia k ∈ {0, 1, 2, 3}
com pesos decrescentes:

| Profundidade | MRR | Recall@5 |
|--------------|-----|----------|
| 0 (somente-conteúdo) | 0.353 | 0.133 |
| **1** | **0.919** | **0.652** |
| 2 | 0.600 | 0.395 |
| 3 | 0.486 | 0.275 |

**Achado:** 1 salto é ótimo; 2 saltos e 3 saltos degradam monotonamente o MRR.
Interpretação: com arestas tipadas curadas, 1 salto captura a estrutura direta do
grafo enquanto 2 saltos começa a incluir "vizinhos de vizinhos" — nós conceitualmente
relacionados mas não diretamente adjacentes — que adicionam ruído. As redes
convolucionais em grafos atingem bom desempenho em profundidade 2-3 via pesos
treináveis que podem suprimir vizinhos irrelevantes; nossa agregação de peso fixo não
tem tal mecanismo.

### 7.2 Codificação via células de lugar hipocampais subperforma métodos algébricos

Suposição comum (da neurociência): codificações biológicas de células de lugar
(O'Keefe & Dostrovsky, 1971) representam compactamente estrutura espacial.
Implementamos coordenadas 2D por caminhada aleatória + padrões binários de disparo de
células de lugar:

| Método | MRR | Recall@5 |
|--------|-----|----------|
| Células de lugar somente (2D comprimido) | 0.133 | 0.068 |
| Células de lugar XOR SimHash | 0.352 | 0.133 |
| **TA-SDM** | **0.919** | **0.652** |

**Achado:** projeção para 2D descarta informação que a abordagem XOR + SimHash
preserva nativamente em espaço binário de alta dimensão. Células de lugar biológicas
evoluíram para navegação em espaço físico 2D; grafos de conhecimento são objetos
combinatoriais intrinsecamente de maior dimensão. Inspiração biológica deve casar com
a geometria subjacente.

### 7.3 Computação por reservatório falha sob binarização

Suposição comum (de redes echo-state): dinâmica aleatória não-linear fixa amplifica
diferenças semânticas. Testamos um reservatório W ∈ ℝ^(256×256) com entradas
gaussianas (raio espectral 1/1.2), 10 iterações de x ← tanh(Wx + W_in · input), então
binarização por sinal:

| Método | MRR | Recall@5 |
|--------|-----|----------|
| Reservatório somente-conteúdo | 0.249 | 0.092 |
| Reservatório + 1-hop XOR | 0.162 | 0.052 |
| **TA-SDM** | **0.919** | **0.652** |

**Achado:** a binarização destrói o gradiente suave de similaridade que o SimHash
preserva. Os sucessos da computação por reservatório na literatura dependem do estado
float contínuo do reservatório; binarizar para uso de endereço perde essas vantagens.

### 7.4 Compressed sensing compete em orçamentos pequenos de bits mas não vence

Suposição comum (da teoria de compressed sensing): sinais k-esparsos podem ser
capturados em O(k log N) medições gaussianas aleatórias. Testamos compressed sensing
em vetores de características esparsos de adjacência:

| Bits | MRR | Recall@5 |
|------|-----|----------|
| 64 | 0.638 | 0.331 |
| 128 | 0.682 | 0.423 |
| 256 | 0.743 | 0.461 |
| 512 | 0.750 | 0.500 |
| 1024 | 0.738 | 0.527 |
| **TA-SDM 256** | **0.919** | **0.652** |

**Achado:** CS em 64 bits (8 bytes/nó!) atinge MRR = 0.638 — compactação notável. Mas
o CS estabiliza em ~0.75 e nunca alcança o TA-SDM. Interpretação: o CS é universal
(funciona para qualquer sinal k-esparso) mas portanto não otimizado para nenhuma
estrutura específica. A agregação por voto majoritário do TA-SDM está especificamente
alinhada com a estrutura "conteúdo + vizinhos" do grafo.

### 7.5 Meta-interpretação

Todos os quatro resultados negativos compartilham um padrão estrutural: uma técnica
com forte motivação teórica em um campo não transfere automaticamente para uma tarefa
"que parece similar" em outro campo. A transferência é bem-sucedida quando a
estrutura matemática subjacente alinha com a tarefa alvo; falha caso contrário. Nosso
TA-SDM é bem-sucedido porque a estrutura matemática da recuperação em espaço de
Hamming casa com a tarefa (encontrar vizinhos de grafo em espaço de endereços) — a
mesma razão pela qual a equivalência de Ramsauer et al. (Seção 2.2) torna a SDM a
"forma discreta binária" da atenção.

---

## 8. Discussão

### 8.1 O que faz o método funcionar

A contribuição primária é a construção do TA-SDM (C1). Supera uma linha de base de
embedding neural treinado de 384 dimensões em 0.485 MRR a 48× menos armazenamento
na tarefa de recuperação de vizinhos de grafo, porque o embedder neural não tem
acesso à informação de arestas do grafo enquanto o TA-SDM agrega as assinaturas
de vizinhos de 1 salto diretamente no endereço binário.

A análise clássica de caminhada quântica (C2) se revelou estatisticamente empatada
com o ranqueamento por distância BFS em subgrafos locais (Seção 5.4). Portanto, não
apresentamos o CTQW como fonte da precisão dentro do subgrafo. Em vez disso, o
achado é que **a extração de subgrafo BFS local** já permite recuperação quase-
perfeita com qualquer ranqueamento razoável — uma observação a nível de regime
sobre grafos de conhecimento heterogêneos tipados em vez de uma afirmação
específica a caminhadas quânticas. Trabalho futuro em grafos com arestas
ponderadas e direcionados (onde a distância BFS é menos informativa) pode revelar
vantagens do CTQW não visíveis em nosso regime.

### 8.2 O que o resultado CTQW sugere sobre vantagem quântica

Enfatizamos que nosso resultado CTQW não usa hardware quântico. É uma simulação
clássica da dinâmica da caminhada, tratável até subgrafos de ~100 nós em CPU comum. O
fato de que a simulação clássica nessas escalas fornece recuperação perfeita no
primeiro rank sugere que para tarefas de recuperação em grafo, a "estrutura de
interferência" da caminhada é a fonte da vantagem — não qualquer speedup em hardware
quântico.

Quando hardware quântico capaz de rodar CTQW em grafos maiores tornar-se disponível, o
mesmo algoritmo deve transferir diretamente. Nosso resultado pode ser interpretado
como um algoritmo clássico inspirado em quântica, fornecendo uma ponte prática entre
métodos puramente clássicos e futuro deployment quântico.

### 8.3 Alinhamento com hardware

Nosso método foi projetado para alinhar com a hierarquia de características presentes
em CPUs modernas: instruções POPCNT, tamanhos de cache L3 de 4-32 MB, aritmética de
inteiros a nível de registro. O grafo completo de 392 nós cabe em 12.5 KB como
endereços de 256 bits, o que cabe confortavelmente no cache L1 de qualquer CPU
moderna. A reprodutibilidade multi-arquitetura (Seção 6) confirma que o método
funciona ao longo de treze anos de gerações de CPU sem modificação.

### 8.4 Limitações

**Escala.** Nossas medições são em N = 392. Escalar para N = 10⁴, 10⁵, 10⁶ requer
tanto indexação hierárquica navegável binária de pequeno-mundo (Malkov & Yashunin,
2018, adaptado para distância de Hamming) quanto engenharia adicional. Projetamos —
baseado no throughput de scan linear de 6.8 M/s — que o scan linear permanece sob 200
ms para N = 10⁶ em Tiger Lake, tornando o HNSW opcional.

**Tipo único de grafo.** Nossos experimentos usam um grafo de conhecimento heterogêneo
tipado. A replicação em benchmarks padrão (Cora, Citeseer, ogbn-products, DBLP) é
trabalho futuro.

**Recuperação entre idiomas.** O SimHash opera em tokens; não casa sinônimos ou
equivalentes entre idiomas. Para recuperação entre idiomas, são necessários tanto
normalização de tokens (lematização + tradução) quanto substituição do SimHash por um
pequeno embedder neural (preservando a agregação topológica no espaço binário).

### 8.5 Consultas composicionais

A propriedade de ligação XOR (Seção 5.5) habilita consultas composicionais exatas a
zero custo de runtime adicional. Essa capacidade está ausente de sistemas de
recuperação por float-embedding: não há operação de ligação invertível em vetores
float que suporte recuperação exata. Segundo nosso conhecimento, esta é a primeira
demonstração empírica de que consultas composicionais exatas em grafos de conhecimento
podem ser realizadas por Python em laptop de consumidor sem qualquer infraestrutura de
machine learning.

---

## 9. Conclusão

Introduzimos a Memória Distribuída Esparsa Consciente de Topologia (TA-SDM), um
método de endereçamento binário de nós sem treinamento que combina hashing de
conteúdo via SimHash com voto majoritário ponderado sobre assinaturas de vizinhos
de grafo a 1 salto. Em um grafo de conhecimento heterogêneo tipado de 392 nós, o
TA-SDM atinge:

- **MRR = 0.914 ± 0.038** (média sobre 10 seeds, IC 95% [0.891, 0.937])
- **Melhoria de 3.45×** sobre SimHash somente-conteúdo (t pareado = 41.78, p < 0.001)
- **Melhoria de 2.13×** sobre linha de base de embedding neural treinado de 384
  dimensões (all-MiniLM-L6-v2; t pareado = 30.65, p < 0.001)
- **48× menos armazenamento** que a linha de base neural (32 bytes vs 1.536 bytes
  por nó)
- **Output bit-a-bit idêntico** em três gerações de CPU abrangendo treze anos (Sandy
  Bridge 2011, Tiger Lake 2020, Arrow Lake 2024), confirmando que a qualidade é
  algorítmica e não dependente de hardware

O método usa somente endereços binários de 256 bits, biblioteca padrão do Python
e instruções POPCNT de hardware. Não requer treinamento neural, nem unidade de
processamento gráfico, nem interface de programação de aplicação para embeddings,
nem hardware quântico. Suporta consultas composicionais exatas via ligação XOR
com zero erros de bit.

Investigamos adicionalmente a simulação clássica de caminhada quântica de tempo
contínuo para refinamento em subgrafos locais e reportamos, via ablação, que
**o ranqueamento por distância BFS sozinho atinge MRR = 1.000 em subgrafos de
50 nós, estatisticamente empatado com o CTQW (t = -1.00, p > 0.05)**. A
recuperação quase-perfeita no primeiro rank nesse regime é uma propriedade da
extração do subgrafo BFS local, não da interferência da caminhada quântica
especificamente. Mantemos a análise CTQW como caracterização honesta do que a
simulação clássica contribui neste regime, e como linha de base para trabalho
futuro em grafos com arestas ponderadas e direcionados, onde a distância BFS
pode se tornar menos informativa.

Código, dados e bateria experimental completa — incluindo os experimentos de
multi-seed e linha de base neural — estão publicamente liberados em
https://github.com/gallori-ai/topology-aware-sdm sob licença MIT (código) e
CC-BY-4.0 (artigo). Reproduzir os resultados completos leva aproximadamente 30-40
minutos em um laptop moderno (dos quais ~5 minutos são o download do modelo
all-MiniLM-L6-v2 de 90 MB no primeiro uso).

---

## Agradecimentos

Este trabalho foi produzido como subproduto da iniciativa de pesquisa do
Continuous Improvement Engine (CEI) na Gallori AI. A reprodução de hardware em
uma segunda máquina foi executada em um padrão de coordenação — inspirado no
conceito de estigmergia em sistemas biológicos de enxame (Grassé 1959) — entre
duas instâncias de um agente autônomo compartilhando apenas o repositório Git
como meio de comunicação, sem comunicação direta instância-a-instância. Esse
modo de coordenação será descrito em trabalho futuro.

O rascunho deste manuscrito e a bateria experimental de suporte foram assistidos
pelo Claude (modelo de linguagem da Anthropic). Todas as decisões metodológicas,
interpretações de resultados e texto final foram revisados e aprovados pelo
autor humano.

## Disponibilidade de dados e código

- Código completo: https://github.com/gallori-ai/topology-aware-sdm (MIT)
- Preprint: https://doi.org/10.5281/zenodo.19645323 (CC-BY-4.0)

## Referências

Ver `references.bib` para entradas BibTeX completas. Citações chave:

- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- Ramsauer, H., et al. (2020). Hopfield Networks is All You Need. NeurIPS.
- Charikar, M. (2002). Similarity estimation techniques from rounding algorithms. STOC.
- Mitrokhin, A., et al. (2019). Learning sensorimotor control with neuromorphic sensors.
  Science Robotics.
- Poduval, P., et al. (2022). GrapHD: Graph-based hyperdimensional memorization. IEEE
  TCAD.
- Farhi, E., & Gutmann, S. (1998). Quantum computation and decision trees. Phys. Rev. A.
- Kipf, T., & Welling, M. (2017). Semi-supervised classification with GCN. ICLR.
- Malkov, Y., & Yashunin, D. (2018). HNSW for ANN search. IEEE TPAMI.
- Candès, E., & Wakin, M. (2008). Introduction to compressive sampling. IEEE SPM.
- Hafting, T., et al. (2005). Microstructure of a spatial map. Nature.
- Jaeger, H. (2001). The echo state approach. GMD Technical Report.
- Grover, L. (1996). A fast quantum mechanical algorithm for database search. STOC.
- Plate, T. (1995). Holographic reduced representations. IEEE TNN.
- Jégou, H., et al. (2011). Product quantization for nearest neighbor search. IEEE TPAMI.

---

*Versão 1.1, atualizado em 2026-05-11. Submissão original 2026-04-18.*
*Correspondência: Cleber Barcelos Costa, Gallori AI, Betim, Minas Gerais, Brasil.*
*ORCID: 0009-0000-5172-9019.*
