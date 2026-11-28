"""Códigos de alimentos (V9001) usados nas flags de pizza e refrigerante.

Cada código traz a descrição do "Cadastro de Produtos do Consumo Alimentar"
oficial e as edições da POF em que ele existe. Quando um código existe nas
duas edições, a descrição é a de 2017-18, com a de 2008-09 entre parênteses
se o texto mudou. Nenhum código troca de significado entre as edições.

A definição final de refrigerante para o alvo ainda não foi decidida: as
flags ficam separadas no parquet e a combinação é feita na modelagem.
"""

from __future__ import annotations

Descricao = tuple[str, tuple[str, ...]]

PIZZA: dict[str, Descricao] = {
    "8500903": ("PIZZA", ("2008-09", "2017-18")),
    "8500914": ("PIZZA CALABREZA", ("2008-09", "2017-18")),
    "8500915": ("PIZZA MUCARELA / MUSSARELA (2008-09: PIZZA MUZZARELA)", ("2008-09", "2017-18")),
    "8500916": ("PIZZA PRESUNTO", ("2008-09", "2017-18")),
    "8500917": ("PIZZA PORTUGUESA", ("2008-09", "2017-18")),
    "8500924": ("PIZZA DE PEPPERONI", ("2017-18",)),
    "8500927": ("PIZZA DE QUEIJOS", ("2017-18",)),
    "8500928": ("PIZZA DE LOMBINHO DEFUMADO", ("2017-18",)),
    "8500929": ("PIZZA DE MUCARELA E PEITO DE PERU", ("2017-18",)),
    "8500930": ("PIZZA NAPOLITANA", ("2017-18",)),
    "8500932": ("PIZZA DE FRANGO COM CATUPIRY", ("2017-18",)),
    "8500934": ("PIZZA DE PEIXE (ATUM, SALMAO, SARDINHA)", ("2017-18",)),
    "8500935": ("PIZZA DE MUCARELA DE BUFALA E TOMATE SECO", ("2017-18",)),
    "8500936": ("PIZZA MARGUERITA (MUCARELA, TOMATE E MANJERICAO)", ("2017-18",)),
    "8500937": ("PIZZA VEGETARIANA", ("2017-18",)),
    "8500939": ("PIZZA HAWAIANA (QUEIJO, PRESUNTO E ABACAXI)", ("2017-18",)),
    "8508017": ("PIZZA PORTUGUESA LIGHT", ("2008-09", "2017-18")),
    # Pizzas light que só existem no cadastro de 2008-09.
    "8508003": ("PIZZA PRONTA LIGHT", ("2008-09",)),
    "8508014": ("PIZZA CALABREZA LIGHT", ("2008-09",)),
    "8508015": ("PIZZA MUZZARELA LIGHT", ("2008-09",)),
    "8508016": ("PIZZA PRESUNTO LIGHT", ("2008-09",)),
}
MINI_PIZZA: dict[str, Descricao] = {
    "6502403": ("MINI PIZZA SEMIPRONTA", ("2008-09",)),
    "6506003": ("MINI PIZZA SEMIPRONTA LIGHT", ("2008-09",)),
}
PIZZA_DOCE: dict[str, Descricao] = {
    "8500938": ("PIZZA DOCE DE QUALQUER SABOR", ("2017-18",)),
}
GUARANA: dict[str, Descricao] = {
    "8200301": ("REFRIGERANTE DE GUARANA TRADICIONAL", ("2008-09", "2017-18")),
    "8200302": ("GUARANA TRADICIONAL", ("2008-09", "2017-18")),
    "8201601": ("REFRIGERANTE DE GUARANA LIGHT", ("2008-09", "2017-18")),
    "8201602": ("GUARANA LIGHT", ("2008-09", "2017-18")),
    "8207401": ("REFRIGERANTE DE GUARANA DIET", ("2008-09", "2017-18")),
    "8207402": ("GUARANA DIET", ("2008-09", "2017-18")),
}
REFRI_GENERICO_SEM_GUARANA: dict[str, Descricao] = {
    "8200101": ("REFRIGERANTE DE COLA TRADICIONAL", ("2008-09", "2017-18")),
    "8200211": ("REFRIGERANTE DE LARANJA TRADICIONAL", ("2017-18",)),
    "8200407": (
        "SPRITE REFRIGERANTE TRADICIONAL (2008-09: SPRIT REFRIGERANTE TRADICIONAL)",
        ("2008-09", "2017-18"),
    ),
    "8200411": ("REFRIGERANTE DE LIMAO TRADICIONAL", ("2017-18",)),
    "8200507": ("REFRIGERANTE DE UVA TRADICIONAL", ("2017-18",)),
    "8200701": ("REFRIGERANTE DE COLA LIGHT", ("2008-09", "2017-18")),
    "8200801": ("REFRIGERANTE DE COLA DIET", ("2008-09", "2017-18")),
    "8201511": ("REFRIGERANTE DE LARANJA DIET", ("2017-18",)),
    "8203501": ("REFRIGERANTE NAO ESPECIFICADO", ("2008-09", "2017-18")),
    "8205402": ("PARAGUAI REFRIGERANTE TRADICIONAL", ("2008-09", "2017-18")),
    "8207501": ("REFRIGERANTE DE LIMAO LIGHT", ("2017-18",)),
    "8207601": ("REFRIGERANTE DE LIMAO DIET", ("2008-09", "2017-18")),
    "8207701": ("REFRIGERANTE DE UVA DIET / LIGHT", ("2017-18",)),
    "8209102": ("PARAGUAI REFRIGERANTE LIGHT", ("2008-09", "2017-18")),
}
REFRI_MARCA: dict[str, Descricao] = {
    "8200102": ("COCA COLA TRADICIONAL", ("2008-09", "2017-18")),
    "8200202": ("FANTA LARANJA TRADICIONAL", ("2008-09", "2017-18")),
    "8200204": ("CRUSH TRADICIONAL", ("2008-09",)),
    "8200208": ("SUKITA TRADICIONAL", ("2008-09", "2017-18")),
    "8200505": ("FANTA UVA TRADICIONAL", ("2008-09", "2017-18")),
    "8200702": ("COCA COLA LIGHT", ("2008-09", "2017-18")),
    "8200902": ("MINUANO TRADICIONAL", ("2008-09", "2017-18")),
    "8201202": ("BIDU TRADICIONAL", ("2008-09", "2017-18")),
    "8201302": ("JAO TRADICIONAL", ("2008-09",)),
    "8201402": ("ALOA TRADICIONAL", ("2008-09",)),
    "8201502": ("FANTA LARANJA LIGHT", ("2008-09", "2017-18")),
    "8201504": ("CRUSH LIGHT", ("2008-09",)),
    "8201508": ("SUKITA LIGHT", ("2008-09",)),
    "8201702": ("XODO DA BAHIA TRADICIONAL", ("2008-09",)),
    "8201802": ("TUBAINA TRADICIONAL", ("2008-09", "2017-18")),
    "8201902": ("TUBAINA LIGHT", ("2008-09", "2017-18")),
    "8207705": ("FANTA UVA LIGHT", ("2008-09", "2017-18")),
    "8207902": ("MINUANO LIGHT", ("2008-09",)),
    "8208202": ("BIDU LIGHT", ("2008-09",)),
    "8208302": ("JAO LIGHT", ("2008-09",)),
    "8208402": ("ALOA LIGHT", ("2008-09",)),
    "8208502": ("XODO DA BAHIA LIGHT", ("2008-09",)),
    "8201017": ("BEBIDA REFRIGERANTE, TIPO H2O", ("2017-18",)),
}
MATE: dict[str, Descricao] = {
    "8201104": ("MATE TRADICIONAL", ("2008-09", "2017-18")),
    "8208104": ("MATE LIGHT", ("2008-09", "2017-18")),
    "8212802": ("MATE DIET", ("2017-18",)),
}
AGUA_TONICA: dict[str, Descricao] = {
    "8204902": ("AGUA TONICA TRADICIONAL", ("2008-09", "2017-18")),
    "8209002": ("AGUA TONICA LIGHT", ("2008-09", "2017-18")),
    "8212902": ("AGUA TONICA DIET", ("2008-09", "2017-18")),
    # Refrigerante de quinino é água tônica, apesar do nome no cadastro.
    "8212906": ("REFRIGERANTE DE QUININO DIETETICO", ("2008-09", "2017-18")),
}

# Lista de refrigerante proposta inicialmente: guaraná + refrigerantes por sabor.
# "Genérico" aqui quer dizer "da lista proposta", não "sem marca": a lista
# inclui Sprite (8200407) e Paraguai (8205402, 8209102). O alvo decidido
# (opção A) usa a união com REFRI_MARCA, então a divisão não afeta o alvo.
REFRI_GENERICO: dict[str, Descricao] = GUARANA | REFRI_GENERICO_SEM_GUARANA

# Nome da flag no parquet -> conjunto de códigos.
# Pizza doce não entra na flag de pizza; as pizzas light entram.
FLAGS: dict[str, frozenset[str]] = {
    "pizza": frozenset(PIZZA),
    "guarana": frozenset(GUARANA),
    "refri_generico": frozenset(REFRI_GENERICO),
    "refri_marca": frozenset(REFRI_MARCA),
    "mini_pizza": frozenset(MINI_PIZZA),
    "pizza_doce": frozenset(PIZZA_DOCE),
    "mate": frozenset(MATE),
    "agua_tonica": frozenset(AGUA_TONICA),
}
