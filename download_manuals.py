import os
import requests

# Map game names (underscores instead of spaces) to their manual URLs
games = {
    "Monopoly": "https://www.hasbro.com/common/instruct/monins.pdf",
    "Dungeons_and_Dragons_5E_SRD": "https://media.wizards.com/2016/downloads/DND/SRD-OGL_V5.1.pdf",
    "UNO": "https://service.mattel.com/instruction_sheets/W2085-UNO.pdf",
    "Scrabble": "https://www.hasbro.com/common/instruct/Scrabble_%282003%29.pdf",
    "Settlers_of_Catan": "https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf",
    "Risk": "https://www.hasbro.com/common/instruct/risk.pdf",
    "Ticket_to_Ride": "https://cdn.1j1ju.com/medias/2c/f9/7f-ticket-to-ride-rulebook.pdf",
    "Pandemic": "https://images-cdn.zmangames.com/us-east-1/filer_public/25/12/251252dd-1338-4f78-b90d-afe073c72363/zm7101_pandemic_rules.pdf",
    "Carcassonne": "https://images.zmangames.com/filer_public/d5/20/d5208d61-8583-478b-a06d-b49fc9cd7aaa/zm7810_carcassonne_rules.pdf",
    "Power_Grid": "https://www.riograndegames.com/wp-content/uploads/2018/12/Power-Grid-Recharged-Rules.pdf",
    "Agricola": "https://cdn.1j1ju.com/medias/dd/16/f5-agricola-rulebook.pdf",
    "Scythe": "https://cdn.1j1ju.com/medias/68/bc/6c-scythe-rulebook.pdf",
    "Wingspan": "https://www.szellemlovas.hu/szabalyok/fesztavEN.pdf",
    "Betrayal_at_House_on_the_Hill": "https://www.qugs.org/rules/r358504.pdf",
    "Puerto_Rico": "https://cdn.1j1ju.com/medias/46/0f/5c-puerto-rico-rulebook.pdf",
    "Terra_Mystica": "https://cdn.1j1ju.com/medias/9c/2c/c8-terra-mystica-rulebook.pdf",
    "7_Wonders": "https://tesera.ru/images/items/8722/7-Wonders-Rulebook-EN.pdf",
    "Dominion": "https://cdn.1j1ju.com/medias/59/e6/c2-dominion-rulebook.pdf",
    "Splendor": "https://cdn.1j1ju.com/medias/7f/91/ba-splendor-rulebook.pdf"
}

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

for game, url in games.items():
    filename = f"{game}.pdf"
    dest = os.path.join(output_dir, filename)
    print(f"Downloading {game} manual...")
    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Saved → {dest}")
    except Exception as e:
        print(f"⚠️ Failed to download {game}: {e}")
