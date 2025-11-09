
from typing import Text, Dict, Any, List, Optional, Tuple
import json
import math
import re
from openai import OpenAI
from CampusCompass.app.config import OPENAI_API_KEY

# ============================================================================
# RICH CAMPUS KNOWLEDGE BASE (Radboud University Nijmegen)
# ----------------------------------------------------------------------------
# Each entry tries to capture what users actually *say* when describing a place.
# Heavy emphasis on visual cues, surroundings, facilities, and nearby landmarks.
# Also includes "discriminators" that help generate friendly yes/no questions.
# ============================================================================

CAMPUS_BUILDINGS: List[Dict[str, Any]] = [
    # --------------------------------------------------------------------
    # SCIENCE CLUSTER
    # --------------------------------------------------------------------
    {
        "name": "Huygensgebouw",
        "aliases": [
            "huygens", "huygens building", "huygensgebouw",
            "science building", "science faculty", "faculty of science",
            "fnwi", "beta", "bèta", "natuurkunde", "wiskunde", "scheikunde",
            "physics", "math", "chemistry", "biology", "computing science",
            "informatics", "toernooiveld", "toernooiveld 1", "toernooiveld 7",
            "science campus hub", "radboud science", "huygens complex", "huygens labs", "Huygens FNWI"
        ],
        "visual": {
            "height": "long, mid-rise wings; sprawling low-to-mid volumes that extend across multiple courtyards with occasional taller stair cores and plant rooms breaking the roofline",
            "materials": ["brick", "glass", "concrete", "aluminium mullions", "metal sunshades", "perforated panels around plant areas"],
            "colors": ["reddish/brown brick", "grey accents", "a lot of glass", "dark grey/black frames", "light concrete soffits"],
            "shapes": ["long wings", "rectilinear lines", "courtyards", "glazed atria connecting wings", "bridging corridors", "recessed entrances with canopies"],
            "unique": [
                "very long elevations", "science posters visible",
                "many lab windows / blinds",
                "FNWI signage on multiple façades",
                "interior study ‘street’ with open study alcoves",
                "fume-hood roof exhausts and screened plant visible from some angles"
            ]
        },
        "environment": {
            "greenery": ["grass fields", "tree-lined lanes", "planted courtyards with benches", "border hedges and pollarded trees along Toernooiveld"],
            "bikes": ["very large outdoor racks", "big bike flows at class changes", "overflow bike parking near side wings and service courts"],
            "traffic": ["multiple bike paths crossing", "pedestrian arteries", "steady delivery vans to lab/service entrances", "students cutting through to Mercator and Gymnasion"],
            "water": ["water features / ditches at Toernooiveld side (varies by area)", "drainage swales near edges of the site"],
            "ground": ["paved plazas", "service roads", "lab entrances", "stamped-concrete walkways with tactile strips at key doors"]
        },
        "facilities": [
            "FNWI labs (physics/chemistry/biology)", "computing science floors",
            "study areas", "lecture rooms", "coffee corners", "canteen/automats",
            "project rooms and tutorial spaces", "makerspaces / instrumentation workshops",
            "quiet study zones along glass façades", "PC halls for exams and practicals",
            "helpdesk/service points for Science students"
        ],
        "people": [
            "science students", "lab coats occasionally", "CS students with laptops",
            "PhD candidates and postdocs", "international researchers/visitors",
            "technicians moving equipment between labs"
        ],
        "sustainability": [
            "daylight via long glass bands", "energy-efficient retrofit elements",
            "zoned ventilation for labs and offices", "automated blinds and heat-gain control in long elevations",
            "focus on re-use/renovation of wings to reduce embodied carbon over time"
        ],
        "nearby": ["Mercator I", "Mercator II", "Mercator III", "Gymnasion", "Radboud Sport Fields", "Toernooiveld", "HFML-FELIX", "Forum", "Elinor Ostromgebouw", "Collegezalencomplex", "Station Heyendaal"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },
    {
        "name": "Mercator I",
        "aliases": ["mercator 1", "mercator i", "mercator", "mercator tower", "mercator office tower", "mercator campus"],
        "visual": {
            "height": "slim tower; a prominent mid-to-high rise relative to nearby low blocks, easily recognizable from Toernooiveld",
            "materials": ["glass", "steel", "aluminium façade elements", "spandrel panels"],
            "colors": ["a lot of glass", "dark frames", "reflective blue/grey tint depending on light"],
            "shapes": ["tall narrow tower", "crisp rectangular footprint", "setback entrance canopy at base"]
        },
        "environment": {
            "greenery": ["grass strips", "avenues", "planted edges with low hedges"],
            "bikes": ["racks at tower base", "additional racks along the avenue connecting to Huygens"],
            "traffic": ["business/startup vibe", "near Toernooiveld", "visitors arriving by bike or on foot; occasional taxis for meetings"]
        },
        "facilities": ["offices", "research", "startups", "meeting rooms and incubator-style floors", "shared reception areas and small breakout lounges"],
        "people": ["researchers", "data/IT companies", "spin-offs and scale-ups", "consultants visiting tenants"],
        "sustainability": ["modern office tower (general)", "high-performance glazing typical for tech offices", "bike-first access with limited surface parking"],
        "nearby": ["Huygensgebouw", "Mercator II", "Mercator III", "Gymnasion", "Elinor Ostromgebouw", "Collegezalencomplex"],
        "discriminators": {
            "is_tower": True, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },
    {
        "name": "Mercator II",
        "aliases": ["mercator 2", "mercator ii", "mercator tower 2", "mercator second tower"],
        "visual": {
            "height": "slim tower; sibling to Mercator I with similar overall proportion and corporate-tech aesthetic",
            "materials": ["glass", "steel", "aluminium framing", "opaque spandrels"],
            "colors": ["a lot of glass", "dark frames", "subtle mirror-like reflections"],
            "shapes": ["tall narrow tower", "clean rectangular massing", "glazed lobby at ground level"]
        },
        "environment": {
            "greenery": ["grass", "trees", "simple planters at the entrance"],
            "bikes": ["racks nearby", "overflow racks along the connecting paths"],
            "traffic": ["business/research vibe", "steady meeting traffic through the day"]
        },
        "facilities": ["offices", "research", "tenant labs/light R&D", "conference rooms"],
        "people": ["researchers", "companies", "university-industry collaboration staff"],
        "sustainability": ["modern tower (general)", "emphasis on public transport and cycling connections", "daylit floorplates with perimeter glazing"],
        "nearby": ["Huygensgebouw", "Mercator I", "Mercator III", "Toernooiveld", "HFML-FELIX"],
        "discriminators": {
            "is_tower": True, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },
    {
        "name": "Mercator III",
        "aliases": ["mercator 3", "mercator iii", "mercator tower 3"],
        "visual": {
            "height": "slim tower; part of the Mercator family with a contemporary high-tech look",
            "materials": ["glass", "steel", "metal panel accents"],
            "colors": ["a lot of glass", "deep-toned frames with variable reflections in sunlight"],
            "shapes": ["tall narrow tower", "rectilinear plan", "glazed base with lobby seating"]
        },
        "environment": {
            "greenery": ["green strips", "rows of young trees along the approach"],
            "bikes": ["racks", "short-stay loops for quick meetings"],
            "traffic": ["office/research", "visitors announced at reception"]
        },
        "facilities": ["offices", "research", "collaboration spaces", "shared facilities with the other Mercator towers"],
        "people": ["researchers", "data/tech staff", "project teams meeting with university groups"],
        "sustainability": ["modern tower (general)", "extensive glazing to maximize daylight", "bike-first commuting culture"],
        "nearby": ["Huygensgebouw", "Toernooiveld", "Mercator I", "Mercator II", "Gymnasion"],
        "discriminators": {
            "is_tower": True, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },

    # --------------------------------------------------------------------
    # MANAGEMENT / LECTURE HUB
    # --------------------------------------------------------------------
    {
        "name": "Elinor Ostromgebouw",
        "aliases": [
            "elinor ostrom", "eos", "eos building",
            "nijmegen school of management", "management building",
            "bedrijfskunde", "economics", "bestuurskunde", "politicologie",
            "NSM building", "Ostrom NSM"
        ],
        "visual": {
            "height": "mid-rise, clean rectangular; a composed series of modern blocks forming a cohesive academic complex",
            "materials": ["glass", "modern facade panels", "light metal trims", "brick accents on some elevations"],
            "colors": ["light/neutral", "a lot of glass", "warm touches near entrances"],
            "shapes": ["rectangular volumes", "crisp lines", "clear main entrance frontage with signage"]
        },
        "environment": {
            "greenery": ["planting and grass", "rows of trees framing pedestrian approaches"],
            "bikes": ["very large bike storages (often full)", "spillover racks along the main path"],
            "traffic": ["flows to/from Collegezalencomplex", "busy at lecture changes", "students congregating near café corners"]
        },
        "facilities": ["NSM (economics/business/public admin)", "lecture rooms", "study areas", "seminar and case-teaching rooms", "quiet study cubicles", "faculty services counters"],
        "people": ["management/econ students", "smart-casual vibe", "policy and admin students between classes", "guest lecturers for business events"],
        "sustainability": ["daylight, energy-efficient envelope (general)", "shading and efficient HVAC typical of modern faculty buildings"],
        "nearby": ["Collegezalencomplex", "Huygensgebouw", "Mercator I", "Forum", "Erasmusgebouw"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": True, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },
    {
        "name": "Collegezalencomplex",
        "aliases": [
            "collegezalencomplex", "college zalen complex", "collegezalen complex",
            "czc", "collegezalen", "lecture hall complex", "cc1", "cc2", "cc3",
            "exam halls", "central lecture complex"
        ],
        "visual": {
            "height": "low sprawling blocks; a series of linked pavilions with clear wayfinding",
            "materials": ["brick", "glass", "concrete canopies at entrances"],
            "colors": ["brick red/brown", "glass", "light interior finishes visible from outside"],
            "shapes": ["low wings", "multiple signed entrances (CC1/CC2/...)", "broad steps and ramps to lobbies"]
        },
        "environment": {
            "greenery": ["grass and trees", "seating edges along paths"],
            "bikes": ["extremely busy racks at peak hours", "short-stay rails near each entrance"],
            "traffic": ["crowds at entrances", "clear CC signage", "queues before large lectures and exams"]
        },
        "facilities": ["big lecture halls (CC1..)", "central exam locations", "overflow rooms for events", "cloakroom spaces during exam weeks"],
        "people": ["mixed faculties during peak hours", "invigilators during exams", "large cohorts gathering between time slots"],
        "sustainability": ["pragmatic lecture halls, renovated areas (general)", "emphasis on flow and efficient occupancy turnover"],
        "nearby": ["Elinor Ostromgebouw", "Huygensgebouw", "Erasmusgebouw", "Forum"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": True, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },

    # --------------------------------------------------------------------
    # HUMANITIES CORE
    # --------------------------------------------------------------------
    {
        "name": "Erasmusgebouw",
        "aliases": [
            "erasmus", "erasmus building", "erasmus tower",
            "humanities", "letteren", "arts", "philosophy",
            "hoge toren", "erasmusplein", "tall tower",
            "arts tower", "humanities tower"
        ],
        "visual": {
            "height": "tall landmark tower; a vertical counterpoint to the low/mid-rise campus fabric, visible from multiple approaches",
            "materials": ["concrete", "glass", "exposed structural elements typical of its era"],
            "colors": ["grey/dark", "light highlights at window bands"],
            "shapes": ["high-rise tower", "rectilinear high-rise slab with repetitive window rhythm"]
        },
        "environment": {
            "greenery": ["Erasmusplein trees", "plaza seating", "planters near ground floor"],
            "bikes": ["racks all around", "overflow towards UB and De Refter"],
            "traffic": ["pedestrian plaza", "bike flows", "near library and cafés", "students meeting on the square between classes"]
        },
        "facilities": ["Arts/Humanities offices and rooms", "lecture rooms", "meeting spaces with views over campus", "departmental admin"],
            "people": ["humanities students", "international students", "lecturers moving between seminars", "visitors to public talks"],
        "sustainability": ["functional tower (general)", "progressive upgrades to interiors and services over time"],
        "nearby": ["Universiteitsbibliotheek", "De Refter", "Cultuurcafé", "Comenius A/B/C", "Aula", "Forum"],
        "discriminators": {
            "is_tower": True, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },
    {
        "name": "Universiteitsbibliotheek",
        "aliases": ["universiteitsbibliotheek", "ub", "library", "bibliotheek", "uni library", "radboud library", "study library"],
        "visual": {
            "height": "mid-size library volumes; stacked reading rooms with generous glazing",
            "materials": ["glass", "stone", "metal trims around large bays"],
            "colors": ["light", "transparent", "warm-toned interiors visible from outside at night"],
            "shapes": ["study halls", "façades with big windows", "clear entrance with security gates"]
        },
        "environment": {
            "greenery": ["trees", "benches", "quiet planted pockets for breaks"],
            "bikes": ["huge bike storage nearby", "additional racks along Erasmusplein edge"],
            "traffic": ["quiet zones inside", "steady student flow outside", "evening study crowd during exam weeks"]
        },
        "facilities": ["study halls", "loan desk", "collections", "silent and group study areas", "PC workstations, printers, lockers", "special collections/reading rooms"],
        "people": ["students with laptops", "book carts", "librarians assisting with databases", "groups revising for exams"],
        "sustainability": ["daylight-heavy", "energy-aware (general)", "encouragement of cycling and walking for access"],
        "nearby": ["Erasmusgebouw", "De Refter", "Cultuurcafé", "Aula", "Comeniusgebouw"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": True, "is_church": False, "near_umc": False
        }
    },
    {
        "name": "De Refter",
        "aliases": ["refter", "canteen", "mensa", "grand café refter", "restaurant refter", "campus canteen"],
        "visual": {
            "height": "low pavilion; a transparent food hall with generous glazing and warm interior finishes",
            "materials": ["glass", "wood/stone", "metal roof edges"],
            "colors": ["light", "inviting", "evening glow from interior lighting"],
            "shapes": ["café/restaurant pavilion", "terrace seating spilling towards the plaza"]
        },
        "environment": {
            "greenery": ["plaza with trees", "informal seating on the edges"],
            "bikes": ["racks around", "short-stay parking for quick lunch runs"],
            "traffic": ["lunch crowds", "trays/queues", "student association lunches and casual meetups"]
        },
        "facilities": ["restaurant/canteen", "many seats", "grab-and-go counters", "coffee points", "microwave corners for self-service"],
        "people": ["mix of students/staff", "study groups taking breaks", "families during open days"],
        "sustainability": ["food service (general)", "reusable dishware where possible and waste-sorting points"],
        "nearby": ["Erasmusgebouw", "Universiteitsbibliotheek", "Cultuurcafé", "Aula", "Erasmusplein"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },
    {
        "name": "Cultuurcafé",
        "aliases": ["cultuurcafe", "cultuur café", "studentencafé", "culture cafe", "student cafe", "campus bar", "culture café"],
        "visual": {
            "height": "low café space; open, welcoming frontage with posters and event boards",
            "materials": ["glass", "wood", "metal canopy trims"],
            "colors": ["warm", "cosy", "dimmed evening ambience during events"],
            "shapes": ["café pavilion", "stage area inside with flexible seating"]
        },
        "environment": {
            "greenery": ["small plaza", "trees providing shade to the terrace"],
            "bikes": ["racks close by", "busy during evening events"],
            "traffic": ["music/borrels/events in evenings", "open-mic nights and association gatherings"]
        },
        "facilities": ["bar", "stage", "seating", "AV setup for performances", "board games and informal study corners during daytime"],
        "people": ["students", "associations", "bands and speakers on event nights", "alumni during reunions"],
        "sustainability": ["hospitality (general)", "encouragement of reusable cups during campus sustainability drives"],
        "nearby": ["De Refter", "Erasmusgebouw", "UB", "Erasmusplein"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },

    # --------------------------------------------------------------------
    # LAW / SOCIAL SCIENCES CLUSTER
    # --------------------------------------------------------------------
    {
        "name": "Maria Montessori-gebouw",
        "aliases": [
            "maria montessori", "montessori", "maria montessori building",
            "social sciences", "sociale wetenschappen", "psychology building",
            "ai building", "gedragswetenschappen", "bsi", "montessoristraat",
            "FSS building", "psych/AI hub"
        ],
        "visual": {
            "height": "mid-size, modern; balanced wings organized around light-filled interiors",
            "materials": ["glass", "brick", "light metal accents", "concrete at plinths"],
            "colors": ["light/reddish brick", "large glass bays", "neutral interior palette"],
            "shapes": ["clean wings", "modern volumes", "clear main entrance with canopy and signage"]
        },
        "environment": {
            "greenery": ["tree rows", "borders", "benches", "quiet sitting pockets along the bike path"],
            "bikes": ["large bike storages", "many students", "overflow during peak lecture changeovers"],
            "traffic": ["bike path along building", "steady flow between Montessori, Spinoza and Grotius"]
        },
        "facilities": ["Faculty of Social Sciences", "Psychology", "AI", "lecture rooms", "study spaces", "lab spaces for behavioural research", "project rooms for group work"],
        "people": ["psych/AI students", "groups with posters", "research participants checking in at labs", "interdisciplinary teams"],
        "sustainability": ["daylight-rich", "modern shell (general)", "focus on indoor environmental quality for long study hours"],
        "nearby": ["Spinozagebouw", "Grotiusgebouw", "Erasmusgebouw", "Forum"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },
    {
        "name": "Spinozagebouw",
        "aliases": ["spinoza", "spinozagebouw", "spinoza building", "BSI building", "behavioural science building"],
        "visual": {
            "height": "mid-size; contemporary, with clear bands of glazing and brick",
            "materials": ["glass", "brick", "metal trims"],
            "colors": ["light/red brick", "glass", "neutral interior tones"],
            "shapes": ["modern volumes", "rational, research-focused floorplates"]
        },
        "environment": {
            "greenery": ["planting/trees", "shaded seating nooks near entrances"],
            "bikes": ["bike storages", "overflow racks along the main path"],
            "traffic": ["flow between Montessori and Grotius", "participants arriving for behavioural studies"]
        },
        "facilities": ["behavioural science/research", "study areas", "specialized labs and observation rooms", "meeting and seminar rooms"],
        "people": ["researchers", "masters students", "participants for experiments", "data collection teams"],
        "sustainability": ["daylight, modern shell (general)", "acoustic treatment in research zones"],
        "nearby": ["Maria Montessori-gebouw", "Grotiusgebouw", "Erasmusgebouw", "Forum"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },
    {
        "name": "Grotiusgebouw",
        "aliases": [
            "grotius", "grotius building", "law building",
            "rights", "faculty of law", "rechten", "rechtsgeleerdheid", "juridische faculteit",
            "law faculty building", "Grotius Law"
        ],
        "visual": {
            "height": "mid-size/modern; composed, crisp architectural language emphasizing transparency",
            "materials": ["glass", "light facade panels", "metal fins for solar control"],
            "colors": ["light", "a lot of glass", "refined neutral palette"],
            "shapes": ["crisp, modern volumes", "clear entrance court with seating"]
        },
        "environment": {
            "greenery": ["trees", "small plazas", "planters defining outdoor seating"],
            "bikes": ["large storages", "short-stay racks near the main door"],
            "traffic": ["bike/foot routes to Montessori/Spinoza", "students in suits on moot court days"]
        },
        "facilities": ["Faculty of Law", "lecture rooms", "study areas", "moot court practice rooms", "advisory/clinic-style spaces"],
        "people": ["law students (sometimes dressed up for moot courts)", "staff and guest lecturers", "international students for comparative law courses"],
        "sustainability": ["energy-efficient modern shell (general)", "daylight and shading carefully balanced"],
        "nearby": ["Maria Montessori-gebouw", "Spinozagebouw", "UB", "Erasmusgebouw"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },

    # --------------------------------------------------------------------
    # CAMPUS HEART / ACADEMIC & ADMIN
    # --------------------------------------------------------------------
    {
        "name": "Aula",
        "aliases": ["aula", "ceremoniezaal", "auditorium", "academiegebouw", "ceremonial hall", "graduation hall"],
        "visual": {
            "height": "ceremonial hall (low/mid); dignified entrance sequence leading to a spacious foyer",
            "materials": ["stone", "glass", "polished interior finishes suitable for ceremonies"],
            "colors": ["formal", "light/dark mix", "warm lighting for events"],
            "shapes": ["hall volumes", "foyer", "broad steps and accessible ramps to main doors"]
        },
        "environment": {
            "greenery": ["planted plazas", "framed views towards campus heart"],
            "bikes": ["racks on edges", "clear approach kept open for ceremony guests"],
            "traffic": ["graduations/ceremonies crowds", "academic processions during formal events"]
        },
        "facilities": ["ceremony/academic hall", "reception areas", "cloakrooms for large gatherings"],
        "people": ["graduates", "families", "staff", "honorary guests during convocations"],
        "sustainability": ["formal venue (general)", "efficient lighting and AV systems for events"],
        "nearby": ["Berchmanianum", "Erasmusgebouw", "UB", "Erasmusplein"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },
    {
        "name": "Berchmanianum",
        "aliases": ["berchmanianum", "klooster", "voormalig klooster", "bestuursgebouw", "main administration building", "former monastery"],
        "visual": {
            "height": "monumental monastery-like; elongated wings around courtyards with historic detailing",
            "materials": ["brick", "ornament", "natural stone details", "timber window frames in historic areas"],
            "colors": ["warm brick", "contrasting light trims", "patinated roofing elements"],
            "shapes": ["cloister corridors", "courtyards", "arched or framed openings reflecting heritage"]
        },
        "environment": {
            "greenery": ["gardens", "old trees", "quiet lawns suitable for formal photos"],
            "bikes": ["fewer racks right at façade", "additional racks along approach roads"],
            "traffic": ["calmer, stately", "administrative visitors and meetings; occasional ceremonies"]
        },
        "facilities": ["administration/board", "meeting rooms", "representative reception rooms", "heritage spaces for special occasions"],
        "people": ["staff", "visitors", "delegations for official meetings"],
        "sustainability": ["heritage conservation (general)", "sensitive retrofits to building services over time"],
        "nearby": ["Heyendaalseweg", "Radboudumc", "Aula"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": True
        }
    },
    {
        "name": "Comeniusgebouw",
        "aliases": [
            "comenius", "comenius a", "comenius b", "comenius c",
            "comeniuslaan", "comenius straat", "comenius a -1", "comenius -1", "kelder comenius",
            "comenius building", "comenius block"
        ],
        "visual": {
            "height": "low to mid, older teaching blocks; corridor-linked wings with a classic campus feel",
            "materials": ["brick", "glass", "concrete", "renovated interior finishes in many sections"],
            "colors": ["red/brown brick", "light interior corridors", "updated door and window frames in parts"],
            "shapes": ["blocks with corridors", "clear stair cores and wayfinding signs for A/B/C"]
        },
        "environment": {
            "greenery": ["tree-lined lanes", "pockets of planting between blocks", "benches along the path"],
            "bikes": ["racks near façades", "overflow along Comeniuslaan during peak hours"],
            "traffic": ["steady flows between rooms", "students moving quickly between A/B/C and Erasmusplein"]
        },
        "facilities": ["teaching rooms", "some basement (-1) areas", "project rooms", "exam and tutorial spaces", "staff rooms and small labs in places"],
        "people": ["varied programmes", "first-year cohorts for foundation courses", "invigilators during assessment periods"],
        "sustainability": ["renovated/functional (general)", "incremental upgrades to lighting and thermal comfort"],
        "nearby": ["Erasmusgebouw", "Collegezalencomplex", "Universiteitsbibliotheek", "Forum"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": True, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },
    {
        "name": "Thomas van Aquinostraat",
        "aliases": ["thomas van aquinostraat", "tva", "tva-gebouw", "aquinostraat", "thomas van aquino", "TVA blocks"],
        "visual": {
            "height": "row of older faculty blocks; linear sequences with a clear street-like edge",
            "materials": ["brick", "concrete lintels", "replacement window systems over the years"],
            "colors": ["dark brick", "contrasting light window bands", "patina from age and renovations"],
            "shapes": ["linear street with education blocks", "repetitive façades defining the academic street"]
        },
        "environment": {
            "greenery": ["tree row", "grassy verges", "occasional planters near entrances"],
            "bikes": ["racks along street", "busy during class changeovers"],
            "traffic": ["steady student flow between blocks", "delivery/service vehicles at times"]
        },
        "facilities": ["mixed educational/transition use", "tutorial rooms", "staff offices where active", "temporary spaces during refurbishments elsewhere"],
        "people": ["students in transit", "classes moving between TVA and central campus"],
        "sustainability": ["varied condition, many renovations (general)", "selective upgrades to extend building life"],
        "nearby": ["Erasmusgebouw", "UB", "De Refter", "Erasmusplein", "Forum"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": True, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },

    # --------------------------------------------------------------------
    # SPIRITUAL / COMMUNITY
    # --------------------------------------------------------------------
    {
        "name": "Studentenkerk",
        "aliases": [
            "studentenkerk", "student church", "kerk", "chapel",
            "erasmuslaan 9a", "campus church", "student chapel"
        ],
        "visual": {
            "height": "modest church/chapel scale; intimate interior volume suited to reflection",
            "materials": ["brick", "glass", "timber doors and detailing"],
            "colors": ["brick", "stained/clear glass", "warm interior tones"],
            "shapes": ["chapel form", "steeple/cross details (modest)", "simple gabled silhouette"]
        },
        "environment": {
            "greenery": ["green borders", "quiet path", "small garden-like spaces around the building"],
            "bikes": ["small racks nearby", "occasional visitors’ bikes during services"],
            "traffic": ["calmer, community visitors", "students attending discussions and gatherings"]
        },
        "facilities": ["chapel/services", "community rooms", "quiet rooms for counseling and meetings"],
        "people": ["students/staff visitors", "community groups", "choirs and discussion groups at events"],
        "sustainability": ["heritage/community venue (general)", "low-impact operation with emphasis on community use"],
        "nearby": ["Erasmusgebouw", "UB", "De Refter", "Erasmusplein"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": True, "near_umc": False
        }
    },

    # --------------------------------------------------------------------
    # UMC CLUSTER
    # --------------------------------------------------------------------
    {
        "name": "Radboudumc",
        "aliases": ["radboudumc", "umc", "hospital", "ziekenhuis", "medical center", "radboud umc", "kliniek", "university medical center"],
        "visual": {
            "height": "large hospital complex with multiple wings; a campus within the campus with recognizable clinical blocks",
            "materials": ["glass", "panels", "steel canopies at entries", "service yard screens"],
            "colors": ["light/grey", "medical signage", "blue/white hospital wayfinding palette"],
            "shapes": ["large blocks", "ambulance bays", "bridged links between wings in some areas"]
        },
        "environment": {
            "greenery": ["planting", "courtyard gardens for patients and staff breaks"],
            "bikes": ["massive parking structures", "overflow racks near outpatient entries"],
            "traffic": ["ambulances", "taxis", "OV hub", "steady patient and visitor flow all day"]
        },
        "facilities": ["hospital", "polyclinics", "research", "education and training spaces for medical students", "cafés and support services"],
        "people": ["healthcare staff", "patients", "visitors", "medical students and residents"],
        "sustainability": ["modern healthcare systems (general)", "continuous upgrades to energy and clinical infrastructure"],
        "nearby": ["Station Heyendaal", "Huize Heyendaal", "Heyendaalseweg", "Berchmanianum"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": True
        }
    },
    {
        "name": "Huize Heyendaal",
        "aliases": [
            "huize heyendaal", "huize", "villa heyendaal", "mansion heyendaal",
            "landgoed heyendaal", "huis heyendaal", "kasteel heyendaal", "estate house"
        ],
        "visual": {
            "height": "villa/mansion (2-3 floors); elegant historic residence with formal façade composition",
            "materials": ["historic brick", "ornaments", "stone trims", "timber sash windows"],
            "colors": ["warm brick", "white window frames", "dark roof tiles"],
            "shapes": ["mansion silhouette", "pitched roof", "symmetrical front with central entrance"]
        },
        "environment": {
            "greenery": ["park-like setting", "old trees", "lawn", "formal driveways and garden borders"],
            "bikes": ["fewer racks at façade", "some parking near the estate entrance"],
            "traffic": ["calmer estate driveway", "event guests and small groups for receptions"]
        },
        "facilities": ["representative rooms", "receptions/ceremonies", "small meetings and special events"],
        "people": ["staff/guests at events", "visitors during open days and heritage tours"],
        "sustainability": ["monument conservation (general)", "careful maintenance of historic fabric"],
        "nearby": ["Radboudumc", "Station Heyendaal", "Heyendaalseweg", "Berchmanianum"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": True,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": True
        }
    },
    {
        "name": "Station Heyendaal",
        "aliases": [
            "station heyendaal", "heyendaal station", "train station",
            "station", "perron", "platform", "sprinter", "trein", "arnhem-nijmegen",
            "nijmegen heyendaal", "heyendaal platforms"
        ],
        "visual": {
            "height": "platforms/stops; open-air platforms with simple canopies and signage",
            "materials": ["platform slabs", "steel canopies", "glass windscreens", "lighting masts"],
            "colors": ["grey platforms", "yellow/blue trains", "standard NS signage palette"],
            "shapes": ["long platforms along tracks", "straight canopies, ramps and stairs to access"]
        },
        "environment": {
            "greenery": ["trees along tracks", "green banks beside the rail line"],
            "bikes": ["large bike parking by station", "short-stay loops near entrances"],
            "traffic": ["trains", "bells", "announcements", "steady peak flows at start and end of lecture days"]
        },
        "facilities": ["sprinters to Nijmegen Centraal/Arnhem", "ticket machines", "sheltered waiting areas", "bus connections nearby"],
        "people": ["commuters", "students with backpacks", "staff heading to UMC and campus"],
        "sustainability": ["OV node", "encourages public transport use across the campus community"],
        "nearby": ["Radboudumc", "Huize Heyendaal", "Heyendaalseweg", "bus stops", "Berchmanianum"],
        "discriminators": {
            "is_tower": False, "has_platforms": True, "is_villa": False,
            "is_lecture_hub": False, "is_station": True, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": True
        }
    },

    # --------------------------------------------------------------------
    # SPORTS
    # --------------------------------------------------------------------
    {
        "name": "Gymnasion",
        "aliases": ["gymnasion", "sportcentrum", "sports centre", "radboud sportcentrum", "sports center", "RSC Gymnasion"],
        "visual": {
            "height": "large sports complex; multiple halls and volumes with generous spans",
            "materials": ["glass", "panelled halls", "metal cladding", "glazed lobbies"],
            "colors": ["light", "glass", "sport branding and signage elements"],
            "shapes": ["big hall volumes", "near outdoor fields", "terrace overlooking paths and courts"]
        },
        "environment": {
            "greenery": ["fields", "trees", "landscaped edges around courts"],
            "bikes": ["busy racks", "people in sportswear", "evening peaks for classes"],
            "traffic": ["runners", "teams clustering", "match days for student sports associations"]
        },
        "facilities": ["fitness", "sports halls", "group classes", "café terrace", "climbing/bouldering areas where available", "changing rooms and equipment rental"],
        "people": ["athletes", "teams", "associations", "PE and recreation staff"],
        "sustainability": ["modern installations (general)", "LED lighting and efficient ventilation for halls"],
        "nearby": ["Huygensgebouw", "Sport Fields", "Toernooiveld", "Mercator towers"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": True,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },

    # --------------------------------------------------------------------
    # EXTRA RESEARCH / FORUM
    # --------------------------------------------------------------------
    {
        "name": "HFML-FELIX",
        "aliases": ["hfml", "felix", "hfml-felix", "high field magnet laboratory", "free electron laser", "HFML FELIX labs"],
        "visual": {
            "height": "technical research facility; low to mid volumes with a purposeful industrial character",
            "materials": ["panels", "industrial elements", "service doors and screened plant areas"],
            "colors": ["neutral/industrial", "functional finishes geared to research operations"],
            "shapes": ["technical volumes", "clipped corners and service yards typical of large equipment buildings"]
        },
        "environment": {
            "greenery": ["limited greenery", "functional planting along boundaries"],
            "bikes": ["smaller racks", "researchers arriving in waves aligned to experiments"],
            "traffic": ["research/technical staff", "deliveries for specialized equipment"]
        },
        "facilities": ["magnet lab", "laser facility", "support workshops and control rooms", "visitor briefing rooms for collaborators"],
        "people": ["researchers", "technicians", "international collaborators for experiments"],
        "sustainability": ["specialized research plant", "energy-intensive systems balanced by targeted efficiency where feasible"],
        "nearby": ["Huygensgebouw", "Toernooiveld", "Mercator towers"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    },
    {
        "name": "Forum",
        "aliases": ["forum", "forum gebouw", "forum building", "forum education building"],
        "visual": {
            "height": "modern mid-size; clear contemporary façade language with ample glazing",
            "materials": ["glass", "panels", "aluminium trims"],
            "colors": ["light/neutral", "transparent lobby zones", "calm interior palette for study"],
            "shapes": ["clean contemporary volumes", "clear main entrance and compact footprint"]
        },
        "environment": {
            "greenery": ["green edges", "small planted strips along paths"],
            "bikes": ["racks nearby", "overflow towards Comenius and Erasmusplein"],
            "traffic": ["steady staff/student movement", "quiet study vibe with short bursts at class changes"]
        },
        "facilities": ["offices/classrooms (varied)", "seminar rooms", "open study corners", "support spaces for teaching"],
        "people": ["students", "staff", "small groups meeting between lectures"],
        "sustainability": ["modern shell (general)", "good daylight and efficient services for teaching spaces"],
        "nearby": ["Comenius", "Erasmusgebouw", "Collegezalencomplex", "Universiteitsbibliotheek"],
        "discriminators": {
            "is_tower": False, "has_platforms": False, "is_villa": False,
            "is_lecture_hub": False, "is_station": False, "is_sports_centre": False,
            "is_library": False, "is_church": False, "near_umc": False
        }
    }
]

# ---------------------------------------------------------------------------
# Alias index for quick scan
# ---------------------------------------------------------------------------
ALIAS_TO_CANONICAL: Dict[str, str] = {}
for b in CAMPUS_BUILDINGS:
    for alias in b["aliases"]:
        ALIAS_TO_CANONICAL[alias.lower().strip()] = b["name"]

KB_INDEX: Dict[str, Dict[str, Any]] = {b["name"]: b for b in CAMPUS_BUILDINGS}


def _contains_any(text: str, words: List[str]) -> bool:
    t = text.lower()
    return any(w.lower() in t for w in words)


def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _cheap_local_match(raw: Optional[Text], restrict_to: Optional[List[str]]) -> Optional[str]:
    """
    Fast path: alias substring -> canonical.
    Respects restrict_to.
    Special: 'heyendaal' alone is ambiguous (station vs. villa) -> None to force disambiguation.
    """
    if not raw:
        return None
    txt = raw.lower()

    # Ambiguous 'heyendaal' without qualifier
    if "heyendaal" in txt and not _contains_any(txt, ["station", "trein", "train", "platform", "perron", "huize", "villa", "mansion"]):
        return None

    hits: List[str] = []
    for alias, canon in ALIAS_TO_CANONICAL.items():
        if alias in txt:
            hits.append(canon)

    if not hits:
        return None

    if restrict_to:
        hits = [h for h in hits if h in restrict_to]
        if not hits:
            return None

    # choose most specific (longest name) if multiple
    hits.sort(key=lambda n: len(n), reverse=True)
    return hits[0]


def _tok_score(text: str, tokens: List[str], w: float) -> float:
    t = text.lower()
    return sum(1 for k in tokens if k and k.lower() in t) * w


# Contextual hint boosts (soft)
NEARBY_HINTS = {
    "near_umc": {
        "triggers": ["near radboudumc", "dichtbij radboudumc", "naast umc", "bij ziekenhuis", "close to hospital"],
        "boost": {
            "Radboudumc": 0.25,
            "Huize Heyendaal": 0.18,
            "Station Heyendaal": 0.10
        }
    }
}


def _special_ambiguous_candidates(raw: str, restrict_to: Optional[List[str]], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Specific patterns with dedicated follow-up (now: 'heyendaal').
    """
    txt = (raw or "").lower()
    if "heyendaal" in txt and not _contains_any(
        txt, ["station", "trein", "train", "platform", "perron", "huize", "villa", "mansion"]
    ):
        cands = ["Station Heyendaal", "Huize Heyendaal"]
        if restrict_to:
            cands = [c for c in cands if c in restrict_to]
            if not cands:
                return None

        other_value = context.get("other_value")
        other_label = context.get("other_slot")
        hint = f" Earlier, your {other_label} was '{other_value}'." if other_value in cands else ""

        return {
            "normalized": "UNKNOWN",
            "confidence": 0.0,
            "candidates": [{"name": n, "confidence": 0.5, "reason": "Ambiguous 'Heyendaal' reference."} for n in cands],
            "followup_question": (
                "Do you mean the train stop 'Station Heyendaal' (platforms/tracks) "
                "or the historic villa 'Huize Heyendaal' on the estate?" + hint
            ),
        }
    return None


def _discriminating_followup_yesno(cands: List[str], context: Dict[str, Any]) -> str:
    """
    Friendlier follow-up: 2–3 short yes/no style questions that distinguish candidates by *features*,
    not listing names. We prefer perceptual cues the user can quickly check.
    """
    def pick(attr: str) -> List[str]:
        return [n for n in cands if KB_INDEX.get(n, {}).get("discriminators", {}).get(attr)]

    ques: List[str] = []

    # High-signal binary cues
    if set(cands) & {"Station Heyendaal"}:
        ques.append("Do you see train tracks or platforms nearby?")
    if set(cands) & {"Huize Heyendaal"}:
        ques.append("Does it look like a historic villa in a park?")

    towerish = pick("is_tower")
    if towerish and len(towerish) < len(cands):
        ques.append("Is it a tall tower-like building?")

    if pick("is_sports_centre"):
        ques.append("Do you see sports halls or lots of people in sportswear?")

    if pick("is_library"):
        ques.append("Are there big study halls or library desks inside?")

    if pick("is_church"):
        ques.append("Does it look like a church or chapel (quiet, with a cross)?")

    if pick("is_lecture_hub"):
        ques.append("Are there clearly marked lecture halls/entrances (CC1/CC2 or many big rooms)?")

    if pick("near_umc"):
        ques.append("Are you right next to the hospital complex (Radboudumc)?")

    # Fall back to general perceptual cues
    if len(ques) < 2:
        ques.extend([
            "Do you see many bike racks clustered at the entrance?",
            "Is the façade mostly brick with long bands of glass?"
        ])

    # Cap to 3 compact questions
    ques = ques[:3]
    return " ".join(q if q.endswith("?") else (q + "?") for q in ques)


def _score_candidates(raw: str, restrict_to: Optional[List[str]], context: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """
    Heuristic scorer:
    - Alias/name hits (heavy)
    - Visual tokens (medium)
    - Environment/people/facilities tokens (light)
    - Nearby landmarks and context hints (light)
    Returns list of (name, score, reason).
    """
    txt = raw.lower()
    results: List[Tuple[str, float, str]] = []

    # Context hint boosts from raw + the other slot value
    boosts: Dict[str, float] = {}
    raw_compound = " ".join([txt, _normalize_text(context.get("other_value"))])
    for hint_key, spec in NEARBY_HINTS.items():
        if any(trig in raw_compound for trig in spec["triggers"]):
            for target_name, add in spec["boost"].items():
                boosts[target_name] = boosts.get(target_name, 0.0) + add

    for b in CAMPUS_BUILDINGS:
        name = b["name"]
        if restrict_to and name not in restrict_to:
            continue

        base = 0.0
        reasons: List[str] = []

        # 1) Alias/name
        alias_hits = sum(1 for a in b["aliases"] if a.lower() in txt)
        if alias_hits:
            base += min(0.65, 0.35 + 0.1 * alias_hits)  # max 0.65
            reasons.append(f"alias match x{alias_hits}")

        # 2) Visual tokens
        vis_tokens = []
        vis = b.get("visual", {})
        vis_tokens += vis.get("materials", [])
        vis_tokens += vis.get("colors", [])
        vis_tokens += vis.get("shapes", [])
        vis_tokens += vis.get("unique", [])
        base += _tok_score(txt, vis_tokens, 0.04)  # up to ~0.4 for many hits

        # 3) Environment/facilities/people
        env = b.get("environment", {})
        ppl = b.get("people", [])
        fac = b.get("facilities", [])
        tokens = []
        for k in ("greenery", "bikes", "traffic", "water", "ground"):
            tokens += env.get(k, [])
        tokens += ppl
        tokens += fac
        base += _tok_score(txt, tokens, 0.02)

        # 4) Nearby keywords
        nearby = b.get("nearby", [])
        for n in nearby:
            if n.lower() in txt:
                base += 0.06
                reasons.append(f"nearby '{n}'")

        # 5) Hint boosts
        if name in boosts:
            base += boosts[name]
            reasons.append(f"context boost (+{boosts[name]:.2f})")

        base = max(0.0, min(0.99, base))
        if base > 0.0:
            results.append((name, base, "; ".join(reasons) or "heuristic match"))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


class LLMController:
    """
    Knowledge-driven normalization with AI fallback.
    Order:
      1) alias match
      2) special ambiguity (Heyendaal)
      3) heuristic scoring (visual/environment/nearby/context)
      4) OpenAI fallback with restricted KB + context, if still unclear
    """

    def __init__(self):
        self.api_key = OPENAI_API_KEY
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY missing. Put it in your .env file or config")
        self.client = OpenAI(api_key=self.api_key)

    def _system_prompt(self, restrict_to: Optional[List[str]], context: Dict[str, Any]) -> str:
        allowed = [b for b in CAMPUS_BUILDINGS if (not restrict_to or b["name"] in restrict_to)]
        kb_json = json.dumps(allowed, ensure_ascii=False, indent=2)
        ctx = json.dumps({
            "target_slot": context.get("target_slot"),
            "other_slot": context.get("other_slot"),
            "other_value": context.get("other_value"),
            "previous_candidates": context.get("previous_candidates") or [],
        }, ensure_ascii=False, indent=2)

        rules = [
            "You are CampusCompass, a navigation assistant for Radboud University Nijmegen.",
            "Your job: map a vague description to EXACTLY ONE canonical building name from canonical_buildings.",
            "If unsure, return UNKNOWN and ask ONE crisp disambiguation follow-up.",
            "Return STRICT JSON only (no extra text).",
            "",
            "Schema:",
            "{",
            '  "normalized": "<canonical OR \\"UNKNOWN\\">",',
            '  "confidence": <float 0..1>,',
            '  "candidates": [ {"name":"<canonical>", "confidence": <float>, "reason":"<short>"} ],',
            '  "followup_question": "<question>"',
            "}",
            "",
            "Helpful mappings:",
            "- science/beta/FNWI → Huygensgebouw",
            "- EOS/management/business/public admin → Elinor Ostromgebouw",
            "- humanities/letteren/arts/tower → Erasmusgebouw",
            "- CC1/CC2/lecture halls → Collegezalencomplex",
            "- station/perron/trein → Station Heyendaal",
            "- law/faculty of law → Grotiusgebouw",
            "- villa/landgoed → Huize Heyendaal (NOT the station)",
            "- near Radboudumc is a hint for Radboudumc *or* Huize Heyendaal",
        ]
        if restrict_to:
            rules += [
                "",
                "IMPORTANT: Restrict-to mode active. Only choose from:",
                ", ".join(restrict_to),
                "If none fits, return UNKNOWN.",
            ]

        return "\n".join(rules + ["", "canonical_buildings =", kb_json, "", "context =", ctx])

    def _friendly_followup(self, ranked: List[Tuple[str, float, str]], context: Dict[str, Any]) -> str:
        cands = [n for (n, _, __) in ranked[:3]]
        return _discriminating_followup_yesno(cands, context)

    def normalize_building(
        self,
        raw: Text,
        top_k: int = 3,
        restrict_to: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        ctx: Dict[str, Any] = context or {}

        # 0) no text
        if raw is None or raw.strip() == "":
            return {
                "normalized": "",
                "confidence": 0.0,
                "candidates": [],
                "followup_question": "What building name or a clear landmark do you see?",
            }

        # 1) direct alias
        direct = _cheap_local_match(raw, restrict_to)
        if direct:
            return {
                "normalized": direct,
                "confidence": 0.92,
                "candidates": [{"name": direct, "confidence": 0.92, "reason": "Matched known alias."}],
                "followup_question": "Is that correct, or do you mean a different building?",
            }

        # 2) special ambiguity (Heyendaal)
        special = _special_ambiguous_candidates(raw, restrict_to, ctx)
        if special:
            return special

        # 3) heuristic ranking
        ranked = _score_candidates(raw, restrict_to, ctx)
        if ranked:
            best_name, best_score, _reason = ranked[0]
            # map raw score -> confidence via S-curve
            conf = float(1.0 / (1.0 + math.exp(-6 * (best_score - 0.55))))
            candidates_payload = [
                {"name": n, "confidence": float(1.0 / (1.0 + math.exp(-6 * (s - 0.55)))),
                 "reason": r}
                for (n, s, r) in ranked[:top_k]
            ]

            if conf >= 0.82:
                return {
                    "normalized": best_name,
                    "confidence": conf,
                    "candidates": candidates_payload,
                    "followup_question": "Is that correct?" if conf >= 0.9 else self._friendly_followup(ranked, ctx),
                }
            else:
                return {
                    "normalized": "UNKNOWN",
                    "confidence": conf,
                    "candidates": candidates_payload,
                    "followup_question": self._friendly_followup(ranked, ctx),
                }

        # 4) AI fallback with KB + context
        system = self._system_prompt(restrict_to, ctx)
        user_payload = {
            "raw_user_text": raw,
            "top_k": top_k,
            "restrict_to": restrict_to or [],
            "instructions": (
                "Infer best canonical building. Use the context to tailor follow-up. "
                "If unsure, return UNKNOWN and ask one crisp discriminating question."
            ),
        }
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
            )
            data = json.loads(response.choices[0].message.content)
        except Exception:
            return {
                "normalized": None,
                "confidence": 0.0,
                "candidates": [],
                "followup_question": "I didn’t catch that. Any nearby signs (faculties), platforms, or a café/park vibe?",
            }

        normalized = data.get("normalized")
        confidence = float(data.get("confidence", 0.0) or 0.0)

        if restrict_to and normalized not in restrict_to and normalized != "UNKNOWN":
            normalized = "UNKNOWN"
            confidence = 0.0

        raw_cands = data.get("candidates") or []
        clean_cands: List[Dict[str, Any]] = []
        for c in raw_cands:
            nm = c.get("name")
            if not nm:
                continue
            if restrict_to and nm not in restrict_to:
                continue
            clean_cands.append(
                {"name": nm, "confidence": float(c.get("confidence", 0.0) or 0.0), "reason": c.get("reason", "")}
            )

        # If the model didn't produce a friendly follow-up, synthesize one from top candidates
        followup = data.get("followup_question")
        if not followup:
            followup = _discriminating_followup_yesno([c["name"] for c in clean_cands], ctx)
        if not followup:
            followup = "Can you name nearby signs, platforms, or a faculty/café?"

        return {
            "normalized": normalized,
            "confidence": confidence,
            "candidates": clean_cands[: top_k],
            "followup_question": followup,
        }
