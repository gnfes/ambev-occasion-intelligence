"""Rótulos legíveis dos valores aceitos pela API."""

TURNOS = {
    "manha": "Manhã (0h às 10h)",
    "almoco": "Almoço (11h às 14h)",
    "tarde": "Tarde (15h às 17h)",
    "noite": "Noite (18h às 23h)",
}
TIPOS_AREA = {
    "capital": "Capital",
    "resto_rm": "Região metropolitana (fora da capital)",
    "interior_urbano": "Interior urbano",
    "rural": "Rural",
}
SEXOS = {"homem": "Homem", "mulher": "Mulher"}
CANAIS = {
    None: "Não informado",
    "delivery": "App de delivery",
    "pizzaria": "Pizzaria do bairro",
    "em_casa": "Consumo em casa",
}
FAIXAS = {"alta": "Alta", "media": "Média", "baixa": "Baixa"}
CORES_FAIXA = {"alta": "green", "media": "orange", "baixa": "gray"}


def classe_de_renda(renda_sm: float) -> str:
    """Mesmo corte do engine (IBGE, renda do domicílio em salários mínimos)."""
    for limite, classe in [(2, "E"), (4, "D"), (10, "C"), (20, "B")]:
        if renda_sm <= limite:
            return classe
    return "A"
