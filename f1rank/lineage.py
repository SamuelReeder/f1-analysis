"""Map Jolpica constructor IDs onto stable team lineages (same factory / entry).

Car performance carries over between seasons within a lineage, so rebrands must
not start a new team. Display names come from the season's own constructor.
"""

LINEAGE = {
    # Brackley
    "honda": "brackley", "brawn": "brackley", "mercedes": "brackley",
    # Faenza
    "toro_rosso": "faenza", "alphatauri": "faenza", "rb": "faenza",
    # Silverstone
    "mf1": "silverstone", "spyker_mf1": "silverstone", "spyker": "silverstone",
    "force_india": "silverstone", "racing_point": "silverstone", "aston_martin": "silverstone",
    # Enstone
    "renault": "enstone", "lotus_f1": "enstone", "alpine": "enstone",
    # Hinwil
    "bmw_sauber": "hinwil", "sauber": "hinwil", "alfa": "hinwil", "audi": "hinwil",
    # Leafield (Lotus Racing -> Caterham)
    "lotus_racing": "leafield", "caterham": "leafield",
    # Banbury (Virgin -> Marussia -> Manor)
    "virgin": "banbury", "marussia": "banbury", "manor": "banbury",
    # Unchanged or short-lived entries
    "red_bull": "red_bull", "ferrari": "ferrari", "mclaren": "mclaren",
    "williams": "williams", "haas": "haas", "cadillac": "cadillac",
    "toyota": "toyota", "super_aguri": "super_aguri", "hrt": "hrt",
}

# Seasons whose technical regulations reset the competitive order.
REGULATION_RESETS = {2009, 2014, 2017, 2022, 2026}


def lineage_of(constructor_id: str) -> str:
    try:
        return LINEAGE[constructor_id]
    except KeyError:
        raise KeyError(f"no lineage for constructor {constructor_id!r}; add it to LINEAGE") from None
