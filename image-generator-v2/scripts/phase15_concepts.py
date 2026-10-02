"""
Phase 1.5 concept list: specific, labeled concepts to deepen the dataset
where Phase 1 is shallow. Every concept becomes one catalogue row; its label
becomes the image's subcategory (and later the start of its caption).

    source "commons"  -> Wikimedia Commons category (and subcategories) found by searching `term`
    source "inat"     -> iNaturalist research-grade photos of the species `term`
    source "megalith" -> Megalith-10m photos whose caption matches the regex `term`
    source "nasa"     -> NASA Image and Video Library search for `term`

Edit freely: add, remove, or change targets. (area, label, source, term, target)
"""

CAR_TARGET = 250
FOOD_TARGET = 250
DRINK_TARGET = 200
SPECIES_TARGET = 300
SCENE_TARGET = 1200
SPACE_TARGET = 300
TECH_TARGET = 200

_CARS = {
    "Ferrari": ["Ferrari 488", "Ferrari F8 Tributo", "Ferrari 458 Italia", "Ferrari SF90 Stradale", "Ferrari Roma",
                "LaFerrari", "Ferrari F40", "Ferrari Testarossa"],
    "Lamborghini": ["Lamborghini Aventador", "Lamborghini Huracán", "Lamborghini Urus", "Lamborghini Countach",
                    "Lamborghini Murciélago"],
    "Porsche": ["Porsche 992", "Porsche 991", "Porsche Cayenne", "Porsche Taycan", "Porsche Macan", "Porsche 918 Spyder"],
    "BMW": ["BMW M3", "BMW M4", "BMW 3 Series", "BMW 5 Series", "BMW X5", "BMW i8", "BMW E30"],
    "Mercedes-Benz": ["Mercedes-Benz G-Class", "Mercedes-Benz S-Class", "Mercedes-Benz C-Class", "Mercedes-AMG GT",
                      "Mercedes-Benz SLS AMG", "Mercedes-Benz 300 SL"],
    "Audi": ["Audi R8", "Audi A4", "Audi Q7", "Audi RS6", "Audi TT"],
    "Tesla": ["Tesla Model S", "Tesla Model 3", "Tesla Model X", "Tesla Model Y", "Tesla Cybertruck"],
    "Toyota": ["Toyota Corolla", "Toyota Camry", "Toyota Supra", "Toyota Land Cruiser", "Toyota Prius", "Toyota Hilux",
               "Toyota AE86"],
    "Honda": ["Honda Civic", "Honda Accord", "Honda NSX", "Honda CR-V"],
    "Nissan": ["Nissan GT-R", "Nissan Skyline", "Nissan 370Z", "Nissan Leaf"],
    "Ford": ["Ford Mustang", "Ford F-150", "Ford GT", "Ford Model T", "Ford Bronco"],
    "Chevrolet": ["Chevrolet Corvette", "Chevrolet Camaro", "Chevrolet Impala"],
    "Volkswagen": ["Volkswagen Beetle", "Volkswagen Golf", "Volkswagen Type 2"],
    "Dodge": ["Dodge Challenger", "Dodge Charger", "Dodge Viper"],
    "Jeep": ["Jeep Wrangler"],
    "Mazda": ["Mazda MX-5", "Mazda RX-7"],
    "Subaru": ["Subaru Impreza WRX"],
    "Land Rover": ["Land Rover Defender", "Range Rover"],
    "Rolls-Royce": ["Rolls-Royce Phantom"],
    "Bentley": ["Bentley Continental GT"],
    "Aston Martin": ["Aston Martin DB5", "Aston Martin Vantage"],
    "McLaren": ["McLaren P1", "McLaren 720S"],
    "Bugatti": ["Bugatti Veyron", "Bugatti Chiron"],
    "Mini": ["Mini Cooper"],
    "Fiat": ["Fiat 500"],
    "Iran Khodro": ["Paykan", "IKCO Samand"],
    "Peugeot": ["Peugeot 405", "Peugeot 206"],
}

_FOODS = [
    # international
    "Pizza Margherita", "Hamburger", "Hot dog", "French fries", "Sushi", "Sashimi", "Ramen", "Pho", "Pad thai",
    "Fried rice", "Dumplings", "Dim sum", "Peking duck", "Spring rolls", "Bibimbap", "Kimchi", "Tacos", "Burrito",
    "Nachos", "Guacamole", "Quesadilla", "Enchiladas", "Paella", "Risotto", "Lasagna", "Spaghetti bolognese",
    "Carbonara", "Gnocchi", "Croissant", "Baguette", "Crêpe", "Macaron", "Éclair", "Tiramisu", "Cheesecake",
    "Brownie", "Doughnut", "Pancakes", "Waffles", "Bagel", "Pretzel", "Fish and chips", "Full breakfast", "Steak",
    "Barbecue ribs", "Fried chicken", "Chicken wings", "Caesar salad", "Greek salad", "Falafel", "Hummus",
    "Shawarma", "Doner kebab", "Baklava", "Biryani", "Butter chicken", "Samosa", "Naan", "Dosa", "Tandoori chicken",
    "Moussaka", "Borscht", "Pierogi", "Goulash", "Wiener schnitzel", "Bratwurst", "Apple pie", "Ice cream", "Gelato",
    "Churros", "Poke", "Tempura", "Onigiri", "Takoyaki", "Miso soup", "Mochi", "Laksa", "Satay", "Nasi goreng",
    "Ceviche", "Empanadas", "Arepa", "Feijoada", "Poutine", "Macaroni and cheese", "Clam chowder", "Lobster roll",
    "Oysters", "Shakshouka", "Tajine", "Couscous", "Jollof rice", "Injera",
    # more cuisines
    "Adobo", "Lechon", "Sinigang", "Halo-halo", "Lumpia",                                  # Filipino
    "Banh mi", "Bun cha", "Goi cuon",                                                      # Vietnamese
    "Tteokbokki", "Korean barbecue", "Japchae",                                            # Korean
    "Tom yum", "Green curry", "Som tam", "Mango sticky rice",                              # Thai
    "Rendang", "Gado-gado",                                                                # Indonesian
    "Okonomiyaki", "Katsu curry", "Udon", "Yakitori",                                      # Japanese
    "Mapo tofu", "Hot pot", "Char siu", "Xiaolongbao",                                     # Chinese
    "Iskender kebab", "Lahmacun", "Pide", "Manti", "Menemen", "Kofta",                     # Turkish
    "Tabbouleh", "Fattoush", "Kibbeh", "Mansaf", "Mezze",                                  # Levantine
    "Doro wat", "Bobotie", "Fufu", "Suya",                                                 # African
    "Lomo saltado", "Pão de queijo", "Brigadeiro", "Chiles en nogada", "Mole (sauce)",     # Latin American
    "Pelmeni", "Blini", "Khachapuri", "Plov",                                              # Russian / Caucasus / Central Asian
    "Smørrebrød", "Swedish meatballs", "Haggis", "Pastel de nata", "Raclette", "Fondue",   # European
    # Persian
    "Chelo Kabab", "Joojeh kabab", "Ghormeh Sabzi", "Fesenjan", "Tahdig", "Tahchin", "Zereshk polo", "Baghali polo",
    "Ash reshteh", "Kashk-e bademjan", "Mirza ghasemi", "Sholeh zard", "Faloodeh", "Abgoosht", "Kuku sabzi",
    "Sangak", "Barbari bread", "Gaz (candy)",
]

_DRINKS = ["Espresso", "Cappuccino", "Latte art", "Tea", "Bubble tea", "Smoothie", "Mojito", "Margarita (cocktail)",
           "Martini (cocktail)", "Red wine", "Beer", "Sake", "Lemonade", "Milkshake", "Doogh"]

# (common name used as the label, scientific name used for the iNaturalist lookup)
_SPECIES = [
    # mammals
    ("Red fox", "Vulpes vulpes"), ("Gray wolf", "Canis lupus"), ("Brown bear", "Ursus arctos"),
    ("Polar bear", "Ursus maritimus"), ("American black bear", "Ursus americanus"), ("Raccoon", "Procyon lotor"),
    ("Red panda", "Ailurus fulgens"), ("Giant panda", "Ailuropoda melanoleuca"), ("Lion", "Panthera leo"),
    ("Tiger", "Panthera tigris"), ("Leopard", "Panthera pardus"), ("Cheetah", "Acinonyx jubatus"),
    ("Snow leopard", "Panthera uncia"), ("Jaguar", "Panthera onca"), ("African elephant", "Loxodonta africana"),
    ("Giraffe", "Giraffa camelopardalis"), ("Plains zebra", "Equus quagga"), ("Hippopotamus", "Hippopotamus amphibius"),
    ("White rhinoceros", "Ceratotherium simum"), ("Western gorilla", "Gorilla gorilla"),
    ("Chimpanzee", "Pan troglodytes"), ("Bornean orangutan", "Pongo pygmaeus"), ("Koala", "Phascolarctos cinereus"),
    ("Eastern grey kangaroo", "Macropus giganteus"), ("Brown-throated sloth", "Bradypus variegatus"),
    ("Eurasian otter", "Lutra lutra"), ("North American beaver", "Castor canadensis"), ("Moose", "Alces alces"),
    ("White-tailed deer", "Odocoileus virginianus"), ("American bison", "Bison bison"),
    ("Dromedary camel", "Camelus dromedarius"), ("Meerkat", "Suricata suricatta"),
    ("European hedgehog", "Erinaceus europaeus"), ("Red squirrel", "Sciurus vulgaris"),
    # birds
    ("Bald eagle", "Haliaeetus leucocephalus"), ("Golden eagle", "Aquila chrysaetos"),
    ("Peregrine falcon", "Falco peregrinus"), ("Barn owl", "Tyto alba"), ("Snowy owl", "Bubo scandiacus"),
    ("Great horned owl", "Bubo virginianus"), ("Greater flamingo", "Phoenicopterus roseus"),
    ("Indian peafowl", "Pavo cristatus"), ("King penguin", "Aptenodytes patagonicus"), ("Toco toucan", "Ramphastos toco"),
    ("Ruby-throated hummingbird", "Archilochus colubris"), ("Common kingfisher", "Alcedo atthis"),
    ("Blue jay", "Cyanocitta cristata"), ("Northern cardinal", "Cardinalis cardinalis"),
    ("American robin", "Turdus migratorius"), ("Mallard", "Anas platyrhynchos"), ("Mute swan", "Cygnus olor"),
    ("Great blue heron", "Ardea herodias"), ("Brown pelican", "Pelecanus occidentalis"),
    ("Atlantic puffin", "Fratercula arctica"), ("Scarlet macaw", "Ara macao"),
    ("Sulphur-crested cockatoo", "Cacatua galerita"),
    # reptiles & amphibians
    ("Green sea turtle", "Chelonia mydas"), ("American alligator", "Alligator mississippiensis"),
    ("Nile crocodile", "Crocodylus niloticus"), ("Veiled chameleon", "Chamaeleo calyptratus"),
    ("Green iguana", "Iguana iguana"), ("Komodo dragon", "Varanus komodoensis"), ("King cobra", "Ophiophagus hannah"),
    ("Ball python", "Python regius"), ("Leopard gecko", "Eublepharis macularius"),
    ("Red-eyed tree frog", "Agalychnis callidryas"), ("Dyeing poison dart frog", "Dendrobates tinctorius"),
    ("Axolotl", "Ambystoma mexicanum"), ("Fire salamander", "Salamandra salamandra"),
    # marine
    ("Bottlenose dolphin", "Tursiops truncatus"), ("Orca", "Orcinus orca"), ("Humpback whale", "Megaptera novaeangliae"),
    ("Great white shark", "Carcharodon carcharias"), ("Giant manta ray", "Mobula birostris"),
    ("Common octopus", "Octopus vulgaris"), ("Clownfish", "Amphiprion ocellaris"), ("Sea otter", "Enhydra lutris"),
    ("Harbor seal", "Phoca vitulina"), ("Walrus", "Odobenus rosmarus"), ("Moon jellyfish", "Aurelia aurita"),
    ("Seahorse", "Hippocampus"),
    # insects
    ("Monarch butterfly", "Danaus plexippus"), ("Honey bee", "Apis mellifera"),
    ("Seven-spot ladybird", "Coccinella septempunctata"), ("European mantis", "Mantis religiosa"),
    ("Common green darner dragonfly", "Anax junius"), ("Stag beetle", "Lucanus cervus"), ("Luna moth", "Actias luna"),
]

# (label, caption regex). Megalith captions are one-sentence scene descriptions;
# matching is limited to the start of the caption (PHASE15_SCENE_MATCH_CHARS)
# and [^.]* keeps both halves of a pattern inside one sentence.
_SCENES = [
    ("city street at night",
     r"\b(street|avenue|alley|road)s?\b[^.]*\b(at night|night-?time|illuminated|lit up)\b|\bnight(-?time)?\b[^.]*\b(street|avenue|alley)s?\b"),
    ("city skyline at night",
     r"\b(skyline|cityscape)\b[^.]*\b(at night|night|illuminated|lit up)\b|\bnight\b[^.]*\b(skyline|cityscape)\b"),
    ("neon signs at night",
     r"\bneon signs?\b|\bneon-lit (streets?|signs?|alleys?|city|storefronts?)\b|\bneon lights?\b[^.]*\b(street|city|signs?|storefronts?|shops?)\b"),
    ("city street in the rain",
     r"\b(rainy|raining|rain-soaked|in the rain)\b[^.]*\b(street|sidewalk|road|city)\b|\b(street|sidewalk|city)\b[^.]*\b(in the rain|rainy|raining|rain-soaked)\b|\bumbrellas?\b[^.]*\b(street|sidewalk)\b"),
    ("snowy city street",
     r"\b(snowy|snow-covered|snowfall)\b[^.]*\b(street|road|city|town)\b|\b(street|road|town)\b[^.]*\b(covered in snow|snowy|snowfall)\b"),
    ("foggy city",
     r"\b(fog|foggy|misty|hazy)\b[^.]*\b(city|street|bridge|skyline|cityscape)\b|\b(city|skyline|bridge)\b[^.]*\b(shrouded in|in the|covered in) (fog|mist)\b"),
    ("city at sunset",
     r"\b(city|skyline|cityscape)\b[^.]*\b(sunset|dusk|twilight|golden hour)\b|\b(sunset|dusk|twilight)\b[^.]*\b(city|skyline|cityscape)\b"),
    ("highway light trails at night", r"\blight trails?\b|\b(long exposure|streaks of light)\b[^.]*\b(traffic|highway|road|cars)\b"),
    ("fireworks over a city", r"\bfireworks?\b[^.]*\b(sky|over|burst|bursting|explode|exploding|display|light up)\b"),
    ("thunderstorm", r"\b(lightning (strikes?|flashes|illuminates|forks)|forks? of lightning|thunderstorm|storm clouds|stormy sky)\b"),
    ("foggy forest", r"\b(fog|foggy|misty)\b[^.]*\b(forest|woods)\b|\b(forest|woods)\b[^.]*\b(fog|mist)\b"),
    ("snowy mountain village", r"\b(snowy|snow-covered)\b[^.]*\b(village|cabins?|chalets?|cottages?)\b"),
    ("rainbow over a landscape",
     r"\b(a|double|full) rainbow\b(?![- ](flag|colou?red|stripe|pattern|shirt|t-shirt|logo|sweater|hat|design|scarf|umbrella|wig|cake|banner|sign|on))[^.]*\b(sky|over|field|landscape|hills|mountains?|horizon|arcs?|stretches)\b"),
    ("aurora borealis",
     r"\baurora borealis\b|\bnorthern lights\b|\b(green|vibrant|colorful|glowing) aurora\b|\baurora (lights up|glows|dances|illuminates|over)\b"),
    ("starry night sky",
     r"\b(starry|star-filled|star-studded) (night )?sky\b(?! (backdrop|pattern|print|mural|wallpaper|design|painting))|\bmilky way\b|\bnight sky\b[^.]*\b(stars|dotted|filled)\b"),
]

# Flickr's image CDN started refusing downloads (403) after the Megalith import,
# so scenes come from these Commons categories instead. Set SCENES_FROM_MEGALITH
# back to True to use the caption regexes above once Flickr serves images again.
SCENES_FROM_MEGALITH = False
# label -> (source, term). "commons" walks that exact Commons category (checked to
# exist); "wsearch" uses Commons full-text search where no clean category exists.
_SCENE_COMMONS = {
    "city street at night": ("commons", "Streets at night"),
    "city skyline at night": ("commons", "Night skylines"),
    "neon signs at night": ("commons", "Neon signs"),
    "rainy city street": ("wsearch", "rainy city street"),
    "rain": ("commons", "Rain"),
    "snowy city street": ("commons", "Snowy streets"),
    "fog and mist": ("commons", "Fog"),
    "city at sunset": ("wsearch", "city skyline sunset"),
    "highway light trails at night": ("commons", "Light trails"),
    "fireworks over a city": ("commons", "Fireworks"),
    "thunderstorm": ("commons", "Lightning"),
    "snowy mountain village": ("wsearch", "snowy mountain village winter"),
    "rainbow over a landscape": ("commons", "Rainbows"),
    "aurora borealis": ("commons", "Aurora borealis"),
    "starry night sky": ("commons", "Night sky"),
}
SCENE_COMMONS_TARGET = 600

_SPACE = [("spiral galaxy", "spiral galaxy"), ("nebula", "nebula"), ("star cluster", "star cluster"),
          ("the Moon", "moon surface"), ("Mars surface", "mars surface"), ("Saturn", "saturn"),
          ("Jupiter", "jupiter"), ("solar eclipse", "solar eclipse"), ("rocket launch", "rocket launch"),
          ("International Space Station", "international space station"), ("astronaut spacewalk", "spacewalk")]

_TECH = ["iPhone", "Samsung Galaxy smartphones", "Laptop computers", "MacBook", "Desktop computers",
         "PlayStation 5", "Xbox Series X", "Nintendo Switch", "Digital single-lens reflex cameras", "Headphones",
         "Smartwatches", "DJI drones", "Virtual reality headsets", "Computer keyboards", "Game controllers",
         "Televisions"]


def concepts() -> list[tuple[str, str, str, str, int]]:
    out = []
    for brand, models in _CARS.items():
        out += [("Transportation & Vehicles", m, "commons", m, CAR_TARGET) for m in models]
    out += [("Food & Cooking", f, "commons", f, FOOD_TARGET) for f in _FOODS]
    out += [("Drinks & Beverages", d, "commons", d, DRINK_TARGET) for d in _DRINKS]
    out += [("Animals & Wildlife", common, "inat", sci, SPECIES_TARGET) for common, sci in _SPECIES]
    if SCENES_FROM_MEGALITH:
        out += [("Cities & Urban Life", label, "megalith", rx, SCENE_TARGET) for label, rx in _SCENES]
    else:
        out += [("Cities & Urban Life", label, source, term, SCENE_COMMONS_TARGET)
                for label, (source, term) in _SCENE_COMMONS.items()]
    out += [("Space & Astronomy", label, "nasa", term, SPACE_TARGET) for label, term in _SPACE]
    out += [("Technology & Electronics", t, "commons", t, TECH_TARGET) for t in _TECH]
    return out


if __name__ == "__main__":
    import collections
    c = concepts()
    by = collections.defaultdict(lambda: [0, 0])
    for area, _, _, _, target in c:
        by[area][0] += 1
        by[area][1] += target
    for area, (n, t) in by.items():
        print(f"{area:<28} {n:>4} concepts  {t:>7,} images")
    print(f"{'TOTAL':<28} {len(c):>4} concepts  {sum(v[1] for v in by.values()):>7,} images")
