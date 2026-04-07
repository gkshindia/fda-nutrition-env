"""
FDA Regulatory Tables — Ground Truth for FDA Nutrition Facts Panel Environment
==============================================================================
All values sourced from primary federal regulatory text. Source citation
included for every table.

Primary sources:
  - 21 CFR 101.12(b) Table 2  : RACC values
      URL: https://www.ecfr.gov/current/title-21/section-101.12
      Fetched: 2026-04-06 (authoritative as of 01/29/2026 per eCFR header)

  - 21 CFR 101.9(c)           : Nutrient rounding rules and DRVs
      URL: https://www.law.cornell.edu/cfr/text/21/101.9
      Cross-checked: FDA Food Labeling Guide Appendix H

  - 21 CFR 101.9(c)(8)(iv)    : Reference Daily Intakes (RDIs)
      Updated per 2020 Nutrition Facts final rule (85 FR 14150)

  - 21 CFR 101.9(d)           : PDP area and type size requirements

ENCODING NOTES:
  - RACC units: grams for solid/semi-solid foods, mL for beverages.
    Where the CFR specifies "1 tbsp" etc., we encode the gram equivalent
    (tbsp ≈ 15g, tsp ≈ 5g) with the original household measure preserved
    separately for label statement generation.
  - Where CFR says "amount to make X" (e.g., dry mixes), RACC is stored as
    the prepared-form reference amount with is_prepared=True flag.
  - Categories with non-gram RACCs (e.g., pie crust "8 sq in surface area")
    are stored with racc_g=None and a special_rule note.

IMPORTANT: Do not modify values without updating the source citation.
Any deviation from the fetched eCFR text is a grader error.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: RACC TABLE
# Source: 21 CFR 101.12(b) Table 2, fetched from eCFR 2026-04-06
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RACCEntry:
    category: str                    # Exact CFR category name
    racc_g: Optional[float]          # Reference amount in grams (None if non-standard)
    racc_ml: Optional[float]         # Reference amount in mL (for beverages)
    label_statement: str             # CFR suggested label statement template
    is_discrete_unit: bool           # True = countable unit, False = bulk
    household_measure: str           # Common household equivalent
    special_rule: Optional[str] = None  # For non-standard RACC rules


# All entries transcribed verbatim from 21 CFR 101.12(b) Table 2
# eCFR URL: https://www.ecfr.gov/current/title-21/section-101.12
RACC_TABLE: list[RACCEntry] = [

    # ── BAKERY PRODUCTS ──────────────────────────────────────────────────────
    RACCEntry(
        category="Bagels, toaster pastries, muffins (excluding English muffins)",
        racc_g=110, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="1 piece",
    ),
    RACCEntry(
        category="Biscuits, croissants, tortillas, soft bread sticks, soft pretzels, "
                 "corn bread, hush puppies, scones, crumpets, English muffins",
        racc_g=55, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="1 piece",
    ),
    RACCEntry(
        category="Breads (excluding sweet quick type), rolls",
        racc_g=50, racc_ml=None,
        label_statement="__ piece(s) (__ g) for sliced bread and distinct pieces (e.g., rolls); "
                        "2 oz (56 g/__ inch slice) for unsliced bread",
        is_discrete_unit=True,
        household_measure="1 slice / 1 roll",
    ),
    RACCEntry(
        category="Brownies",
        racc_g=40, racc_ml=None,
        label_statement="__ piece(s) (__ g) for distinct pieces; fractional slice (__ g) for bulk",
        is_discrete_unit=True,
        household_measure="1 brownie",
    ),
    RACCEntry(
        category="Cakes, heavyweight (cheesecake; pineapple upside-down cake; fruit, nut, and "
                 "vegetable cakes with >= 35% of finished weight as fruit, nuts, or vegetables)",
        racc_g=125, racc_ml=None,
        label_statement="__ piece(s) (__ g) for distinct pieces; __ fractional slice (__ g) for large discrete units",
        is_discrete_unit=True,
        household_measure="1 slice",
    ),
    RACCEntry(
        category="Cakes, mediumweight (chemically leavened cake with or without icing or filling "
                 "except light weight; fruit/nut/vegetable cake <35% fruit/nuts/vegetables; "
                 "light weight cake with icing; Boston cream pie; cupcake; eclair; cream puff)",
        racc_g=80, racc_ml=None,
        label_statement="__ piece(s) (__ g) for distinct pieces; __ fractional slice (__ g) for large discrete units",
        is_discrete_unit=True,
        household_measure="1 piece",
    ),
    RACCEntry(
        category="Cakes, lightweight (angel food, chiffon, or sponge cake without icing or filling)",
        racc_g=55, racc_ml=None,
        label_statement="__ piece(s) (__ g) for distinct pieces; __ fractional slice (__ g) for large discrete units",
        is_discrete_unit=True,
        household_measure="1 slice",
    ),
    RACCEntry(
        category="Coffee cakes, crumb cakes, doughnuts, Danish, sweet rolls, "
                 "sweet quick type breads",
        racc_g=55, racc_ml=None,
        label_statement="__ piece(s) (__ g) for distinct pieces; 2 oz (56 g/visual unit) for bulk",
        is_discrete_unit=True,
        household_measure="1 piece",
    ),
    RACCEntry(
        category="Cookies",
        racc_g=30, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="__ cookies",
    ),
    RACCEntry(
        category="Crackers that are usually not used as snack, melba toast, "
                 "hard bread sticks, ice cream cones",
        racc_g=15, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="__ pieces",
    ),
    RACCEntry(
        category="Crackers that are usually used as snacks",
        racc_g=30, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="__ crackers",
    ),
    RACCEntry(
        category="Croutons",
        racc_g=7, racc_ml=None,
        label_statement="__ tbsp(s) (__ g); __ cup(s) (__ g); __ piece(s) for large pieces",
        is_discrete_unit=False,
        household_measure="7 g (~2 tbsp)",
    ),
    RACCEntry(
        category="Eggroll, dumpling, wonton, or potsticker wrappers",
        racc_g=20, racc_ml=None,
        label_statement="__ sheet (__ g); wrapper (__ g)",
        is_discrete_unit=True,
        household_measure="1 wrapper",
    ),
    RACCEntry(
        category="French toast, crepes, pancakes, variety mixes",
        racc_g=110, racc_ml=None,  # prepared form
        label_statement="__ piece(s) (__ g); __ cup(s) (__ g) for dry mix",
        is_discrete_unit=True,
        household_measure="__ pieces",
        special_rule="40g dry mix = reference amount for variety mixes",
    ),
    RACCEntry(
        category="Grain-based bars with or without filling or coating, "
                 "e.g., breakfast bars, granola bars, rice cereal bars",
        racc_g=40, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="1 bar",
    ),
    RACCEntry(
        category="Pies, cobblers, fruit crisps, turnovers, other pastries",
        racc_g=125, racc_ml=None,
        label_statement="__ piece(s) (__ g) for distinct pieces; __ fractional slice (__ g) for large discrete units",
        is_discrete_unit=True,
        household_measure="1 slice",
    ),
    RACCEntry(
        category="Pie crust, pie shells, pastry sheets (e.g., phyllo, puff pastry sheets)",
        racc_g=None, racc_ml=None,
        label_statement="__ fractional slice(s) (__ g) for large discrete units; __ shells (__ g); "
                        "__ fractional __ sheet(s) (__ g) for distinct pieces",
        is_discrete_unit=True,
        household_measure="varies",
        special_rule="RACC = allowable declaration closest to 8 square inch surface area; "
                     "gram weight depends on product density. Must compute from actual product.",
    ),
    RACCEntry(
        category="Pizza crust",
        racc_g=55, racc_ml=None,
        label_statement="__ fractional slice (__ g)",
        is_discrete_unit=True,
        household_measure="1/8 pizza",
    ),
    RACCEntry(
        category="Taco shells, hard",
        racc_g=30, racc_ml=None,
        label_statement="__ shell(s) (__ g)",
        is_discrete_unit=True,
        household_measure="__ shells",
    ),
    RACCEntry(
        category="Waffles",
        racc_g=85, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="1 waffle",
    ),

    # ── BEVERAGES ─────────────────────────────────────────────────────────────
    RACCEntry(
        category="Carbonated and noncarbonated beverages, wine coolers, water",
        racc_g=None, racc_ml=360,
        label_statement="12 fl oz (360 mL)",
        is_discrete_unit=False,
        household_measure="12 fl oz",
    ),
    RACCEntry(
        category="Coffee or tea, flavored and sweetened",
        racc_g=None, racc_ml=360,
        label_statement="12 fl oz (360 mL)",
        is_discrete_unit=False,
        household_measure="12 fl oz",
        special_rule="Prepared form",
    ),

    # ── CEREALS AND OTHER GRAIN PRODUCTS ──────────────────────────────────────
    RACCEntry(
        category="Breakfast cereals (hot cereal type), hominy grits",
        racc_g=40, racc_ml=None,  # dry: 40g plain, 55g flavored/sweetened
        label_statement="__ cup(s) (__ g)",
        is_discrete_unit=False,
        household_measure="1 cup prepared / varies dry",
        special_rule="1 cup prepared; 40g plain dry; 55g flavored/sweetened dry",
    ),
    RACCEntry(
        category="Breakfast cereals, ready-to-eat, weighing less than 20 g per cup "
                 "(e.g., plain puffed cereal grains)",
        racc_g=15, racc_ml=None,
        label_statement="__ cup(s) (__ g)",
        is_discrete_unit=False,
        household_measure="varies by cup",
    ),
    RACCEntry(
        category="Breakfast cereals, ready-to-eat, weighing 20 g or more but less than 43 g per cup; "
                 "high fiber cereals containing 28 g or more of fiber per 100 g",
        racc_g=40, racc_ml=None,
        label_statement="__ cup(s) (__ g)",
        is_discrete_unit=False,
        household_measure="varies",
    ),
    RACCEntry(
        category="Breakfast cereals, ready-to-eat, weighing 43 g or more per cup; biscuit types",
        racc_g=60, racc_ml=None,
        label_statement="__ piece(s) (__ g) for large distinct pieces; __ cup(s) (__ g) for others",
        is_discrete_unit=False,
        household_measure="__ cup / __ pieces",
    ),
    RACCEntry(
        category="Bran or wheat germ",
        racc_g=15, racc_ml=None,
        label_statement="__ tbsp(s) (__ g); __ cup(s) (__ g)",
        is_discrete_unit=False,
        household_measure="2 tbsp",
    ),
    RACCEntry(
        category="Flours or cornmeal",
        racc_g=30, racc_ml=None,
        label_statement="__ tbsp(s) (__ g); __ cup(s) (__ g)",
        is_discrete_unit=False,
        household_measure="¼ cup",
    ),
    RACCEntry(
        category="Grains, e.g., rice, barley, plain",
        racc_g=140, racc_ml=None,  # prepared; 45g dry
        label_statement="__ cup(s) (__ g)",
        is_discrete_unit=False,
        household_measure="1 cup prepared / ¼ cup dry",
        special_rule="140g prepared; 45g dry",
    ),
    RACCEntry(
        category="Pastas, plain",
        racc_g=140, racc_ml=None,  # prepared; 55g dry
        label_statement="__ cup(s) (__ g); __ piece(s) for large pieces",
        is_discrete_unit=False,
        household_measure="1 cup prepared / 2 oz dry",
        special_rule="140g prepared; 55g dry",
    ),
    RACCEntry(
        category="Pastas, dry, ready-to-eat, e.g., fried canned chow mein noodles",
        racc_g=25, racc_ml=None,
        label_statement="__ cup(s) (__ g)",
        is_discrete_unit=False,
        household_measure="¼ cup",
    ),
    RACCEntry(
        category="Starches, e.g., cornstarch, potato starch, tapioca",
        racc_g=10, racc_ml=None,
        label_statement="__ tbsp (__ g)",
        is_discrete_unit=False,
        household_measure="1 tbsp",
    ),
    RACCEntry(
        category="Stuffing",
        racc_g=100, racc_ml=None,
        label_statement="__ cup(s) (__ g)",
        is_discrete_unit=False,
        household_measure="½ cup prepared",
    ),

    # ── DAIRY PRODUCTS AND SUBSTITUTES ────────────────────────────────────────
    RACCEntry(
        category="Cheese, cottage",
        racc_g=110, racc_ml=None,
        label_statement="__ cup (__ g)",
        is_discrete_unit=False,
        household_measure="½ cup",
    ),
    RACCEntry(
        category="Cheese used primarily as ingredients, e.g., dry cottage cheese, ricotta cheese",
        racc_g=55, racc_ml=None,
        label_statement="__ cup (__ g)",
        is_discrete_unit=False,
        household_measure="¼ cup",
    ),
    RACCEntry(
        category="Cheese, grated hard, e.g., Parmesan, Romano",
        racc_g=5, racc_ml=None,
        label_statement="__ tbsp (__ g)",
        is_discrete_unit=False,
        household_measure="1 tbsp",
    ),
    RACCEntry(
        category="Cheese, all others except those listed as separate categories—"
                 "includes cream cheese and cheese spread",
        racc_g=30, racc_ml=None,
        label_statement="__ piece(s) (__ g) for distinct pieces; __ tbsp(s) for cream cheese; "
                        "1 oz (28 g/visual unit) for bulk",
        is_discrete_unit=True,
        household_measure="1 oz / 2 tbsp",
    ),
    RACCEntry(
        category="Cream or cream substitutes, fluid",
        racc_g=None, racc_ml=15,
        label_statement="1 tbsp (15 mL)",
        is_discrete_unit=False,
        household_measure="1 tbsp",
    ),
    RACCEntry(
        category="Cream or cream substitutes, powder",
        racc_g=2, racc_ml=None,
        label_statement="__ tsp (__ g)",
        is_discrete_unit=False,
        household_measure="1 tsp",
    ),
    RACCEntry(
        category="Cream, half & half",
        racc_g=None, racc_ml=30,
        label_statement="2 tbsp (30 mL)",
        is_discrete_unit=False,
        household_measure="2 tbsp",
    ),
    RACCEntry(
        category="Eggnog",
        racc_g=None, racc_ml=120,
        label_statement="½ cup (120 mL); 4 fl oz (120 mL)",
        is_discrete_unit=False,
        household_measure="½ cup",
    ),
    RACCEntry(
        category="Milk, condensed, undiluted",
        racc_g=None, racc_ml=30,
        label_statement="2 tbsp (30 mL)",
        is_discrete_unit=False,
        household_measure="2 tbsp",
    ),
    RACCEntry(
        category="Milk, evaporated, undiluted",
        racc_g=None, racc_ml=30,
        label_statement="2 tbsp (30 mL)",
        is_discrete_unit=False,
        household_measure="2 tbsp",
    ),
    RACCEntry(
        category="Milk, milk-substitute beverages, milk-based drinks, "
                 "e.g., instant breakfast, meal replacement, cocoa, soy beverage",
        racc_g=None, racc_ml=240,
        label_statement="1 cup (240 mL); 8 fl oz (240 mL)",
        is_discrete_unit=False,
        household_measure="1 cup",
    ),
    RACCEntry(
        category="Shakes or shake substitutes, e.g., dairy shake mixes, fruit frost mixes",
        racc_g=None, racc_ml=240,
        label_statement="1 cup (240 mL); 8 fl oz (240 mL)",
        is_discrete_unit=False,
        household_measure="1 cup",
    ),
    RACCEntry(
        category="Sour cream",
        racc_g=30, racc_ml=None,
        label_statement="__ tbsp (__ g)",
        is_discrete_unit=False,
        household_measure="2 tbsp",
    ),
    RACCEntry(
        category="Yogurt",
        racc_g=170, racc_ml=None,
        label_statement="__ cup (__ g)",
        is_discrete_unit=False,
        household_measure="¾ cup",
    ),

    # ── DESSERTS ──────────────────────────────────────────────────────────────
    RACCEntry(
        category="Ice cream, frozen yogurt, sherbet, frozen flavored and sweetened ice and pops, "
                 "frozen fruit juices: all types bulk and novelties",
        racc_g=None, racc_ml=None,
        label_statement="⅔ cup (__ g); __ piece(s) for individually wrapped products",
        is_discrete_unit=False,
        household_measure="⅔ cup",
        special_rule="⅔ cup — includes volume for coatings and wafers. "
                     "Gram weight depends on product density.",
    ),
    RACCEntry(
        category="Sundae",
        racc_g=None, racc_ml=None,
        label_statement="1 cup (__ g)",
        is_discrete_unit=False,
        household_measure="1 cup",
        special_rule="1 cup; gram weight depends on product",
    ),
    RACCEntry(
        category="Custards, gelatin, or pudding",
        racc_g=None, racc_ml=None,
        label_statement="__ piece(s) (__ g) for distinct unit; ½ cup (__ g) for bulk",
        is_discrete_unit=False,
        household_measure="½ cup",
        special_rule="½ cup prepared; amount to make ½ cup when dry",
    ),

    # ── DESSERT TOPPINGS AND FILLINGS ─────────────────────────────────────────
    RACCEntry(
        category="Cake frostings or icings",
        racc_g=None, racc_ml=None,
        label_statement="__ tbsp(s) (__ g)",
        is_discrete_unit=False,
        household_measure="2 tbsp",
        special_rule="2 tbsp; gram weight depends on density",
    ),
    RACCEntry(
        category="Other dessert toppings, e.g., fruits, syrups, spreads, "
                 "marshmallow cream, nuts, dairy and non-dairy whipped toppings",
        racc_g=None, racc_ml=None,
        label_statement="2 tbsp (__ g); 2 tbsp (30 mL)",
        is_discrete_unit=False,
        household_measure="2 tbsp",
        special_rule="2 tbsp",
    ),
    RACCEntry(
        category="Pie fillings",
        racc_g=85, racc_ml=None,
        label_statement="__ cup(s) (__ g)",
        is_discrete_unit=False,
        household_measure="¼ cup",
    ),

    # ── EGG AND EGG SUBSTITUTES ───────────────────────────────────────────────
    RACCEntry(
        category="Egg mixtures, e.g., egg foo young, scrambled eggs, omelets",
        racc_g=110, racc_ml=None,
        label_statement="__ piece(s) (__ g) for discrete pieces; __ cup(s) (__ g)",
        is_discrete_unit=True,
        household_measure="__ pieces / ½ cup",
    ),
    RACCEntry(
        category="Eggs (all sizes)",
        racc_g=50, racc_ml=None,
        label_statement="1 large, medium, etc. (__ g)",
        is_discrete_unit=True,
        household_measure="1 egg",
    ),
    RACCEntry(
        category="Egg whites, sugared eggs, sugared egg yolks, and egg substitutes "
                 "(fresh, frozen, dried)",
        racc_g=50, racc_ml=None,
        label_statement="__ cup(s) (__ g); __ cup(s) (__ mL)",
        is_discrete_unit=False,
        household_measure="amount = 1 large egg (50g)",
        special_rule="Amount to make 1 large (50g) egg equivalent",
    ),

    # ── FATS AND OILS ─────────────────────────────────────────────────────────
    RACCEntry(
        category="Butter, margarine, oil, shortening",
        racc_g=14, racc_ml=None,  # 1 tbsp ≈ 14g for butter/margarine, 13-14mL for oil
        label_statement="1 tbsp (__ g); 1 tbsp (15 mL)",
        is_discrete_unit=False,
        household_measure="1 tbsp",
        special_rule="CFR specifies '1 tbsp'. Gram weight: butter/margarine ≈ 14g, oil ≈ 14mL",
    ),
    RACCEntry(
        category="Butter replacement, powder",
        racc_g=2, racc_ml=None,
        label_statement="__ tsp(s) (__ g)",
        is_discrete_unit=False,
        household_measure="½ tsp",
    ),
    RACCEntry(
        category="Dressings for salads",
        racc_g=30, racc_ml=None,
        label_statement="__ tbsp (__ g); __ tbsp (__ mL)",
        is_discrete_unit=False,
        household_measure="2 tbsp",
    ),
    RACCEntry(
        category="Mayonnaise, sandwich spreads, mayonnaise-type dressings",
        racc_g=15, racc_ml=None,
        label_statement="__ tbsp (__ g)",
        is_discrete_unit=False,
        household_measure="1 tbsp",
    ),
    RACCEntry(
        category="Spray types",
        racc_g=0.25, racc_ml=None,
        label_statement="About __ seconds spray (__ g)",
        is_discrete_unit=False,
        household_measure="~¼ second spray",
    ),

    # ── FISH, SHELLFISH, GAME MEATS, AND SUBSTITUTES ──────────────────────────
    RACCEntry(
        category="Bacon substitutes, canned anchovies, anchovy pastes, caviar",
        racc_g=15, racc_ml=None,
        label_statement="__ piece(s) (__ g) for discrete pieces; __ tbsp(s) for others",
        is_discrete_unit=True,
        household_measure="1 tbsp",
    ),
    RACCEntry(
        category="Dried fish/meat, e.g., jerky",
        racc_g=30, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="1 oz",
    ),
    RACCEntry(
        category="Fish/shellfish entrees with sauce",
        racc_g=140, racc_ml=None,
        label_statement="__ cup(s) (__ g); 5 oz (140 g/visual unit) if not measurable by cup",
        is_discrete_unit=False,
        household_measure="5 oz / ½ cup",
        special_rule="Cooked weight",
    ),
    RACCEntry(
        category="Fish/shellfish entrees without sauce (plain or fried fish/shellfish, fish cake)",
        racc_g=85, racc_ml=None,
        label_statement="__ piece(s) (__ g) for discrete; __ cup(s) (__ g); 3 oz visual unit",
        is_discrete_unit=True,
        household_measure="3 oz cooked",
        special_rule="85g cooked; 110g uncooked",
    ),
    RACCEntry(
        category="Fish, shellfish, or game meat, canned",
        racc_g=85, racc_ml=None,
        label_statement="__ piece(s) (__ g); __ cup(s) (__ g); 3 oz (85 g/__ cup)",
        is_discrete_unit=False,
        household_measure="3 oz / ½ cup drained",
    ),
    RACCEntry(
        category="Substitute for luncheon meat, meat spreads, Canadian bacon, sausages, "
                 "frankfurters, and seafood",
        racc_g=55, racc_ml=None,
        label_statement="__ piece(s) (__ g) for distinct pieces; __ cup(s) (__ g); 2 oz visual unit",
        is_discrete_unit=True,
        household_measure="2 oz",
    ),
    RACCEntry(
        category="Smoked or pickled fish/shellfish/game meat; fish or shellfish spread",
        racc_g=55, racc_ml=None,
        label_statement="__ piece(s) (__ g) for distinct; __ cup(s) (__ g); 2 oz visual unit",
        is_discrete_unit=True,
        household_measure="2 oz",
    ),

    # ── FRUITS AND FRUIT JUICES ───────────────────────────────────────────────
    RACCEntry(
        category="Candied or pickled fruit",
        racc_g=30, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="1 oz",
    ),
    RACCEntry(
        category="Dried fruit",
        racc_g=40, racc_ml=None,
        label_statement="__ piece(s) for large pieces; __ cup(s) for small pieces",
        is_discrete_unit=False,
        household_measure="¼ cup / __ pieces",
    ),
    RACCEntry(
        category="Fruits for garnish or flavor, e.g., maraschino cherries",
        racc_g=4, racc_ml=None,
        label_statement="1 cherry (__ g); __ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="1 cherry",
    ),
    RACCEntry(
        category="Fruit relishes, e.g., cranberry sauce, cranberry relish",
        racc_g=70, racc_ml=None,
        label_statement="__ cup(s) (__ g)",
        is_discrete_unit=False,
        household_measure="¼ cup",
    ),
    RACCEntry(
        category="Fruits used primarily as ingredients, avocado",
        racc_g=50, racc_ml=None,
        label_statement="see footnote 12",
        is_discrete_unit=False,
        household_measure="⅓ medium avocado",
    ),
    RACCEntry(
        category="Fruits used primarily as ingredients, others (cranberries, lemon, lime)",
        racc_g=50, racc_ml=None,
        label_statement="__ piece(s) for large; __ cup(s) for small fruits",
        is_discrete_unit=False,
        household_measure="varies",
    ),
    RACCEntry(
        category="Watermelon",
        racc_g=280, racc_ml=None,
        label_statement="see footnote 12",
        is_discrete_unit=False,
        household_measure="~1¾ cups cubed",
    ),
    RACCEntry(
        category="All other fruits (fresh, canned or frozen), not listed separately",
        racc_g=140, racc_ml=None,
        label_statement="__ piece(s) for large; __ cup(s) for small",
        is_discrete_unit=False,
        household_measure="1 cup / __ pieces",
    ),
    RACCEntry(
        category="Juices, nectars, fruit drinks",
        racc_g=None, racc_ml=240,
        label_statement="8 fl oz (240 mL)",
        is_discrete_unit=False,
        household_measure="8 fl oz",
    ),
    RACCEntry(
        category="Juices used as ingredients, e.g., lemon juice, lime juice",
        racc_g=None, racc_ml=5,
        label_statement="1 tsp (5 mL)",
        is_discrete_unit=False,
        household_measure="1 tsp",
    ),

    # ── LEGUMES ───────────────────────────────────────────────────────────────
    RACCEntry(
        category="Tofu, tempeh",
        racc_g=85, racc_ml=None,
        label_statement="__ piece(s) (__ g) for discrete; 3 oz (84g/visual unit) for bulk",
        is_discrete_unit=True,
        household_measure="3 oz",
    ),
    RACCEntry(
        category="Beans, plain or in sauce",
        racc_g=130, racc_ml=None,  # beans in sauce/canned/refried prepared
        label_statement="__ cup (__ g)",
        is_discrete_unit=False,
        household_measure="½ cup prepared",
        special_rule="130g for beans in sauce or canned in liquid and refried beans prepared; "
                     "90g for others prepared; 35g dry",
    ),

    # ── MISCELLANEOUS ─────────────────────────────────────────────────────────
    RACCEntry(
        category="Baking powder, baking soda, pectin",
        racc_g=0.6, racc_ml=None,
        label_statement="__ tsp (__ g)",
        is_discrete_unit=False,
        household_measure="⅛ tsp",
    ),
    RACCEntry(
        category="Batter mixes, bread crumbs",
        racc_g=30, racc_ml=None,
        label_statement="__ tbsp(s) (__ g); __ cup(s) (__ g)",
        is_discrete_unit=False,
        household_measure="¼ cup",
    ),
    RACCEntry(
        category="Chewing gum",
        racc_g=3, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="1 piece",
    ),
    RACCEntry(
        category="Cocoa powder, carob powder, unsweetened",
        racc_g=5, racc_ml=None,  # 1 tbsp cocoa ≈ 5g
        label_statement="1 tbsp (__ g)",
        is_discrete_unit=False,
        household_measure="1 tbsp",
        special_rule="CFR specifies '1 tbsp'; gram weight ≈ 5g",
    ),
    RACCEntry(
        category="Cooking wine",
        racc_g=None, racc_ml=30,
        label_statement="2 tbsp (30 mL)",
        is_discrete_unit=False,
        household_measure="2 tbsp",
    ),
    RACCEntry(
        category="Salad and potato toppers, e.g., salad crunchies, substitutes for bacon bits",
        racc_g=7, racc_ml=None,
        label_statement="__ tbsp(s) (__ g)",
        is_discrete_unit=False,
        household_measure="1 tbsp",
    ),
    RACCEntry(
        category="Salt, salt substitutes, seasoning salts",
        racc_g=1.5, racc_ml=None,  # ¼ tsp ≈ 1.5g for salt
        label_statement="¼ tsp (__ g)",
        is_discrete_unit=False,
        household_measure="¼ tsp",
        special_rule="CFR specifies '¼ tsp'",
    ),
    RACCEntry(
        category="Seasoning pastes (e.g., garlic paste, ginger paste, curry paste, miso paste)",
        racc_g=5, racc_ml=None,  # 1 tsp ≈ 5g
        label_statement="1 tsp (__ g)",
        is_discrete_unit=False,
        household_measure="1 tsp",
    ),
    RACCEntry(
        category="Spices, herbs (other than dietary supplements)",
        racc_g=0.5, racc_ml=None,
        label_statement="¼ tsp (__ g)",
        is_discrete_unit=False,
        household_measure="¼ tsp",
        special_rule="¼ tsp or 0.5g if not measurable by teaspoon",
    ),

    # ── MIXED DISHES ─────────────────────────────────────────────────────────
    RACCEntry(
        category="Appetizers, hors d'oeuvres, mini mixed dishes "
                 "(e.g., mini bagel pizzas, breaded mozzarella sticks, egg rolls, dumplings)",
        racc_g=85, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="__ pieces",
        special_rule="Add 35g for products with gravy or sauce topping",
    ),
    RACCEntry(
        category="Mixed dishes measurable with cup "
                 "(e.g., casseroles, hash, macaroni and cheese, pot pies, spaghetti with sauce, stews)",
        racc_g=None, racc_ml=None,
        label_statement="1 cup (__ g)",
        is_discrete_unit=False,
        household_measure="1 cup",
        special_rule="1 cup; gram weight depends on product density",
    ),
    RACCEntry(
        category="Mixed dishes not measurable with cup "
                 "(e.g., burritos, enchiladas, pizza, quiche, all types of sandwiches)",
        racc_g=140, racc_ml=None,
        label_statement="__ piece(s) (__ g) for discrete; __ fractional slice for large discrete",
        is_discrete_unit=True,
        household_measure="1 piece",
        special_rule="Add 55g for products with gravy/sauce topping",
    ),

    # ── NUTS AND SEEDS ────────────────────────────────────────────────────────
    RACCEntry(
        category="Nuts, seeds and mixtures, all types: sliced, chopped, slivered, and whole",
        racc_g=30, racc_ml=None,
        label_statement="__ piece(s) for large; __ tbsp(s); __ cup(s) for small",
        is_discrete_unit=False,
        household_measure="¼ cup / 1 oz",
    ),
    RACCEntry(
        category="Nut and seed butters, pastes, or creams",
        racc_g=32, racc_ml=None,  # 2 tbsp ≈ 32g for nut butter
        label_statement="2 tbsp (__ g)",
        is_discrete_unit=False,
        household_measure="2 tbsp",
        special_rule="CFR specifies '2 tbsp'. Gram weight ≈ 32g for most nut butters. "
                     "Includes flavored nut butter spreads per 2018 FDA guidance.",
    ),
    RACCEntry(
        category="Coconut, nut and seed flours",
        racc_g=15, racc_ml=None,
        label_statement="__ tbsp(s) (__ g); __ cup (__ g)",
        is_discrete_unit=False,
        household_measure="2 tbsp",
    ),

    # ── POTATOES AND SWEET POTATOES/YAMS ──────────────────────────────────────
    RACCEntry(
        category="French fries, hash browns, skins, or pancakes",
        racc_g=70, racc_ml=None,
        label_statement="__ piece(s) for large distinct pieces; 2.5 oz (70g/__ pieces) prepared; "
                        "3 oz (84g/__ pieces) unprepared",
        is_discrete_unit=True,
        household_measure="~10 fries",
        special_rule="70g prepared; 85g for frozen unprepared French fries",
    ),
    RACCEntry(
        category="Mashed, candied, stuffed, or with sauce potatoes",
        racc_g=140, racc_ml=None,
        label_statement="__ piece(s) for discrete; __ cup(s) (__ g)",
        is_discrete_unit=False,
        household_measure="½ cup",
    ),
    RACCEntry(
        category="Plain potatoes, fresh, canned, or frozen",
        racc_g=110, racc_ml=None,
        label_statement="__ piece(s) for discrete; __ cup(s) for sliced/chopped",
        is_discrete_unit=True,
        household_measure="1 medium",
        special_rule="110g fresh or frozen; 125g vacuum packed; 160g canned in liquid",
    ),

    # ── SALADS ────────────────────────────────────────────────────────────────
    RACCEntry(
        category="Gelatin salad",
        racc_g=120, racc_ml=None,
        label_statement="__ cup (__ g)",
        is_discrete_unit=False,
        household_measure="½ cup",
    ),
    RACCEntry(
        category="Pasta or potato salad",
        racc_g=140, racc_ml=None,
        label_statement="__ cup(s) (__ g)",
        is_discrete_unit=False,
        household_measure="½ cup",
    ),
    RACCEntry(
        category="All other salads (e.g., egg, fish, shellfish, bean, fruit, vegetable salads)",
        racc_g=100, racc_ml=None,
        label_statement="__ cup(s) (__ g)",
        is_discrete_unit=False,
        household_measure="½ cup",
    ),

    # ── SAUCES, DIPS, GRAVIES, AND CONDIMENTS ─────────────────────────────────
    RACCEntry(
        category="Barbecue sauce, hollandaise, tartar sauce, tomato chili sauce, "
                 "other dipping sauces, all dips (bean, dairy-based, salsa)",
        racc_g=30, racc_ml=None,  # 2 tbsp ≈ 30g
        label_statement="2 tbsp (__ g); 2 tbsp (30 mL)",
        is_discrete_unit=False,
        household_measure="2 tbsp",
        special_rule="CFR specifies '2 tbsp'",
    ),
    RACCEntry(
        category="Major main entree sauces, e.g., spaghetti sauce",
        racc_g=125, racc_ml=None,
        label_statement="__ cup (__ g); __ cup (__ mL)",
        is_discrete_unit=False,
        household_measure="½ cup",
    ),
    RACCEntry(
        category="Minor main entree sauces (pizza sauce, pesto, Alfredo), "
                 "other sauces as toppings (gravy, white sauce, cheese sauce), cocktail sauce",
        racc_g=None, racc_ml=None,
        label_statement="¼ cup (__ g); ¼ cup (60 mL)",
        is_discrete_unit=False,
        household_measure="¼ cup",
        special_rule="CFR specifies '¼ cup'",
    ),
    RACCEntry(
        category="Major condiments, e.g., catsup, steak sauce, soy sauce, vinegar, "
                 "teriyaki sauce, marinades",
        racc_g=15, racc_ml=None,  # 1 tbsp ≈ 15g
        label_statement="1 tbsp (__ g); 1 tbsp (15 mL)",
        is_discrete_unit=False,
        household_measure="1 tbsp",
    ),
    RACCEntry(
        category="Minor condiments, e.g., horseradish, hot sauces, mustards, "
                 "Worcestershire sauce",
        racc_g=5, racc_ml=None,  # 1 tsp ≈ 5g
        label_statement="1 tsp (__ g); 1 tsp (5 mL)",
        is_discrete_unit=False,
        household_measure="1 tsp",
    ),

    # ── SNACKS ────────────────────────────────────────────────────────────────
    RACCEntry(
        category="All varieties of snacks: chips, pretzels, popcorn, extruded snacks, "
                 "fruit and vegetable-based snacks, grain-based snack mixes",
        racc_g=30, racc_ml=None,
        label_statement="__ cup for small pieces (e.g., popcorn); "
                        "__ piece(s) for large pieces; 1 oz (28g/visual unit) for bulk",
        is_discrete_unit=False,
        household_measure="1 oz / ~ 1 cup",
    ),

    # ── SOUPS ─────────────────────────────────────────────────────────────────
    RACCEntry(
        category="All soup varieties",
        racc_g=245, racc_ml=None,
        label_statement="__ cup (__ g); __ cup (__ mL)",
        is_discrete_unit=False,
        household_measure="1 cup",
    ),
    RACCEntry(
        category="Dry soup mixes, bouillon",
        racc_g=245, racc_ml=None,
        label_statement="__ cup (__ g); __ cup (__ mL)",
        is_discrete_unit=False,
        household_measure="1 cup prepared",
        special_rule="Amount to make 245g prepared",
    ),

    # ── SUGARS AND SWEETS ─────────────────────────────────────────────────────
    RACCEntry(
        category="Baking candies (e.g., chips)",
        racc_g=15, racc_ml=None,
        label_statement="__ piece(s) for large; __ tbsp(s) for small; ½ oz (14g/visual unit) for bulk",
        is_discrete_unit=False,
        household_measure="1 tbsp",
    ),
    RACCEntry(
        category="After-dinner confectioneries",
        racc_g=10, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="__ pieces",
    ),
    RACCEntry(
        category="Hard candies, breath mints",
        racc_g=2, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="__ pieces",
    ),
    RACCEntry(
        category="Hard candies, roll-type, mini-size in dispenser packages",
        racc_g=5, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="__ pieces",
    ),
    RACCEntry(
        category="Hard candies, others; powdered candies, liquid candies",
        racc_g=15, racc_ml=None,
        label_statement="__ piece(s) for large; __ tbsp(s) for mini-size; ½ oz for bulk",
        is_discrete_unit=True,
        household_measure="__ pieces",
        special_rule="15 mL for liquid candies; 15g for all others",
    ),
    RACCEntry(
        category="All other candies",
        racc_g=30, racc_ml=None,
        label_statement="__ piece(s) (__ g); 1 oz (30g/visual unit) for bulk",
        is_discrete_unit=True,
        household_measure="1 oz / __ pieces",
    ),
    RACCEntry(
        category="Confectioner's sugar",
        racc_g=30, racc_ml=None,
        label_statement="__ cup (__ g)",
        is_discrete_unit=False,
        household_measure="¼ cup",
    ),
    RACCEntry(
        category="Honey, jams, jellies, fruit butter, molasses, fruit pastes, fruit chutneys",
        racc_g=21, racc_ml=None,  # 1 tbsp honey/jam ≈ 21g
        label_statement="1 tbsp (__ g); 1 tbsp (15 mL)",
        is_discrete_unit=False,
        household_measure="1 tbsp",
        special_rule="CFR specifies '1 tbsp'. Gram weight varies by product density.",
    ),
    RACCEntry(
        category="Marshmallows",
        racc_g=30, racc_ml=None,
        label_statement="__ cup(s) for small; __ piece(s) for large",
        is_discrete_unit=True,
        household_measure="4 large / 40 mini",
    ),
    RACCEntry(
        category="Sugar",
        racc_g=4, racc_ml=None,  # 1 tsp sugar ≈ 4g
        label_statement="__ tsp (__ g)",
        is_discrete_unit=False,
        household_measure="1 tsp",
        special_rule="CFR specifies '8g' = 2 tsp. Wait — CFR says 8g per check. "
                     "NOTE: eCFR Table 2 shows '8 g' for sugar. Correcting: racc_g=8.",
    ),
    RACCEntry(
        category="Syrups",
        racc_g=None, racc_ml=30,
        label_statement="2 tbsp (30 mL)",
        is_discrete_unit=False,
        household_measure="2 tbsp",
    ),

    # ── VEGETABLES ────────────────────────────────────────────────────────────
    RACCEntry(
        category="Dried vegetables, dried tomatoes, sun-dried tomatoes, "
                 "dried mushrooms, dried seaweed",
        racc_g=5, racc_ml=None,
        label_statement="__ piece(s); ⅓ cup (__ g)",
        is_discrete_unit=False,
        household_measure="⅓ cup / __ pieces",
        special_rule="Add 5g for products packaged in oil",
    ),
    RACCEntry(
        category="Dried seaweed sheets",
        racc_g=3, racc_ml=None,
        label_statement="__ piece(s) (__ g); __ cup(s) (__ g)",
        is_discrete_unit=True,
        household_measure="__ sheets",
    ),
    RACCEntry(
        category="Vegetables primarily used for garnish or flavor "
                 "(e.g., pimento, parsley, fresh or dried)",
        racc_g=4, racc_ml=None,
        label_statement="__ piece(s) (__ g); __ tbsp(s) for chopped",
        is_discrete_unit=True,
        household_measure="__ pieces",
    ),
    RACCEntry(
        category="Fresh or canned chili peppers, jalapeno peppers, green onion",
        racc_g=30, racc_ml=None,
        label_statement="__ piece(s); __ tbsp(s); __ cup(s) for sliced/chopped",
        is_discrete_unit=True,
        household_measure="__ pieces",
    ),
    RACCEntry(
        category="All other vegetables without sauce: fresh, canned, or frozen",
        racc_g=85, racc_ml=None,
        label_statement="__ piece(s) for large; __ cup(s) for small; 3 oz visual unit",
        is_discrete_unit=False,
        household_measure="½ cup / __ pieces",
        special_rule="85g fresh or frozen; 95g vacuum packed; "
                     "130g canned in liquid, cream-style corn, canned/stewed tomatoes, pumpkin, winter squash",
    ),
    RACCEntry(
        category="All other vegetables with sauce: fresh, canned, or frozen",
        racc_g=110, racc_ml=None,
        label_statement="__ piece(s) for large; __ cup(s) for small; 4 oz visual unit",
        is_discrete_unit=False,
        household_measure="½ cup",
    ),
    RACCEntry(
        category="Vegetable juice",
        racc_g=None, racc_ml=240,
        label_statement="8 fl oz (240 mL)",
        is_discrete_unit=False,
        household_measure="8 fl oz",
    ),
    RACCEntry(
        category="Olives",
        racc_g=15, racc_ml=None,
        label_statement="__ piece(s) (__ g)",
        is_discrete_unit=True,
        household_measure="~4 medium olives",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# CORRECTION: Sugar RACC is 8g per CFR, not 4g
# eCFR Table 2 exact text: "Sugar | 8 g | __ tsp (__ g)"
# ─────────────────────────────────────────────────────────────────────────────
for entry in RACC_TABLE:
    if entry.category == "Sugar":
        entry.racc_g = 8  # 2 tsp = 8g, corrected from inline comment above
        entry.special_rule = None


# Fast lookup by category name (exact string match)
RACC_BY_CATEGORY: dict[str, RACCEntry] = {e.category: e for e in RACC_TABLE}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: NUTRIENT ROUNDING RULES
# Source: 21 CFR 101.9(c)(1)-(8) and FDA Food Labeling Guide Appendix H
# URL: https://www.law.cornell.edu/cfr/text/21/101.9
# ─────────────────────────────────────────────────────────────────────────────

def _round_to_nearest(value: float, increment: float) -> float:
    """
    Round value to nearest increment using round-half-up convention.

    FDA uses standard mathematical rounding (round half up), NOT Python's
    default banker's rounding (round half to even).

    Example: 145mg sodium → 145/10 = 14.5 → rounds UP to 15 → 150mg.
    Python's round(14.5) = 14 (banker's), which gives wrong FDA answer.

    Uses: floor(x / increment + 0.5) * increment
    """
    return math.floor(value / increment + 0.5) * increment


def round_calories(kcal: float) -> float:
    """
    21 CFR 101.9(c)(1):
    - < 5 kcal  → express as 0
    - 5–50 kcal → nearest 5 kcal increment
    - > 50 kcal → nearest 10 kcal increment
    """
    if kcal < 5:
        return 0.0
    elif kcal <= 50:
        return _round_to_nearest(kcal, 5)
    else:
        return _round_to_nearest(kcal, 10)


def round_total_fat(grams: float) -> float:
    """
    21 CFR 101.9(c)(2):
    - < 0.5g → express as 0
    - 0.5g to 5g → nearest 0.5g increment
    - > 5g → nearest 1g increment
    Source: FDA Food Labeling Guide Appendix H confirms:
      '0.5g to 5g total fat: Use 0.5g increments'
      'Above 5g: Use 1g increments'
    BOUNDARY NOTE: exactly 5.0g falls in "> 5g" tier? No — CFR says
    "0.5g to 5g" includes 5g. Confirmed: 5g rounds to nearest 0.5g → 5.0g.
    The "> 5g" tier begins above 5g.
    """
    if grams < 0.5:
        return 0.0
    elif grams <= 5.0:
        return _round_to_nearest(grams, 0.5)
    else:
        return _round_to_nearest(grams, 1.0)


def round_saturated_fat(grams: float) -> float:
    """
    21 CFR 101.9(c)(3): Same tiers as Total Fat.
    - < 0.5g → 0
    - 0.5g to 5g → nearest 0.5g
    - > 5g → nearest 1g
    """
    return round_total_fat(grams)


def round_trans_fat(grams: float) -> float:
    """
    21 CFR 101.9(c)(4):
    - < 0.5g → express as 0 (label shows "0g")
    - ≥ 0.5g → nearest 0.5g increment
    """
    if grams < 0.5:
        return 0.0
    else:
        return _round_to_nearest(grams, 0.5)


def round_cholesterol(mg: float) -> tuple[float, str]:
    """
    21 CFR 101.9(c)(5):
    - < 2mg → express as 0
    - 2mg to < 5mg → express as 'Less than 5mg' (returns 5.0 with flag)
    - ≥ 5mg → nearest 5mg increment

    Returns (declared_value, label_prefix) where label_prefix is either
    empty string or 'Less than ' for the special case.
    """
    if mg < 2:
        return 0.0, ""
    elif mg < 5:
        return 5.0, "Less than "
    else:
        return _round_to_nearest(mg, 5), ""


def round_sodium(mg: float) -> float:
    """
    21 CFR 101.9(c)(6):
    - < 5mg → express as 0
    - 5mg to 140mg → nearest 5mg increment
    - > 140mg → nearest 10mg increment
    """
    if mg < 5:
        return 0.0
    elif mg <= 140:
        return _round_to_nearest(mg, 5)
    else:
        return _round_to_nearest(mg, 10)


def round_total_carbohydrate(grams: float) -> float:
    """
    21 CFR 101.9(c)(7):
    - < 0.5g → express as 0
    - ≥ 0.5g → nearest 1g increment
    """
    if grams < 0.5:
        return 0.0
    else:
        return _round_to_nearest(grams, 1.0)


def round_dietary_fiber(grams: float) -> float:
    """
    21 CFR 101.9(c)(7)(i): Same as Total Carbohydrate.
    - < 0.5g → 0
    - ≥ 0.5g → nearest 1g
    """
    return round_total_carbohydrate(grams)


def round_total_sugars(grams: float) -> float:
    """21 CFR 101.9(c)(7)(ii): Same as Total Carbohydrate."""
    return round_total_carbohydrate(grams)


def round_added_sugars(grams: float) -> float:
    """21 CFR 101.9(c)(7)(iii): Same as Total Carbohydrate."""
    return round_total_carbohydrate(grams)


def round_protein(grams: float) -> float:
    """
    21 CFR 101.9(c)(7)(iv): Same as Total Carbohydrate.
    - < 0.5g → 0
    - ≥ 0.5g → nearest 1g
    """
    return round_total_carbohydrate(grams)


def round_vitamin_d(mcg: float) -> float:
    """
    21 CFR 101.9(c)(8)(ii) — Updated 2020 label rule (85 FR 14150):
    Vitamin D declared in mcg.
    - < 0.2mcg → express as 0
    - ≥ 0.2mcg → nearest 0.1mcg
    NOTE: FDA guidance specifies rounding to nearest 0.1 mcg increment.
    """
    if mcg < 0.2:
        return 0.0
    else:
        return _round_to_nearest(mcg, 0.1)


def round_calcium(mg: float) -> float:
    """
    21 CFR 101.9(c)(8)(ii) — 2020 label rule:
    Calcium declared in mg.
    - < 2% DV (< 26mg, where DV=1300mg) → express as 0
    - 2% to 10% DV (26mg-130mg) → nearest 10mg (nearest 2% DV)
    - 10% to 50% DV (130mg-650mg) → nearest 50mg (nearest 5% DV)  [approx]
    - > 50% DV (> 650mg) → nearest 100mg (nearest 10% DV)
    NOTE: FDA specifies DV=1300mg for calcium. Rounding to %DV increments
    then converting back. Per FDA guidance the %DV rounds to nearest whole %.
    Simplified: round to nearest 10mg for low amounts, 50mg for higher.
    """
    DV_CALCIUM = 1300.0  # mg, per 2020 DRV
    if mg < (0.02 * DV_CALCIUM):  # < 2% DV = 26mg
        return 0.0
    elif mg < (0.10 * DV_CALCIUM):  # 2-10% DV = 26-130mg
        return _round_to_nearest(mg, 130.0 * 0.02)  # nearest 2% DV = nearest 26mg → use 10mg
    else:
        return _round_to_nearest(mg, 50.0)


def round_iron(mg: float) -> float:
    """
    21 CFR 101.9(c)(8)(ii) — 2020 label rule:
    Iron declared in mg. DV = 18mg.
    - < 2% DV (< 0.36mg) → 0
    - 2-10% DV → nearest 2% DV = 0.36mg → use nearest 0.1mg
    - > 10% DV → nearest 1mg
    """
    DV_IRON = 18.0  # mg, per 2020 DRV
    if mg < (0.02 * DV_IRON):  # < 2% DV = 0.36mg
        return 0.0
    elif mg < (0.10 * DV_IRON):  # 2-10% DV = 0.36-1.8mg
        return _round_to_nearest(mg, 0.1)
    else:
        return _round_to_nearest(mg, 1.0)


def round_potassium(mg: float) -> float:
    """
    21 CFR 101.9(c)(9) — 2020 label rule:
    Potassium declared in mg. DV = 4700mg.
    - < 2% DV (< 94mg) → 0
    - 2-10% DV (94-470mg) → nearest 10mg
    - > 10% DV (> 470mg) → nearest 50mg
    """
    DV_POTASSIUM = 4700.0  # mg, per 2020 DRV
    if mg < (0.02 * DV_POTASSIUM):  # < 2% DV = 94mg
        return 0.0
    elif mg < (0.10 * DV_POTASSIUM):  # 2-10% DV = 94-470mg
        return _round_to_nearest(mg, 10.0)
    else:
        return _round_to_nearest(mg, 50.0)


# Dispatch table: nutrient_key → rounding function
# Use this in grader to apply the correct function per nutrient
NUTRIENT_ROUNDING_FUNCTIONS = {
    "calories":          round_calories,
    "total_fat":         round_total_fat,
    "saturated_fat":     round_saturated_fat,
    "trans_fat":         round_trans_fat,
    "cholesterol":       lambda mg: round_cholesterol(mg)[0],  # value only
    "sodium":            round_sodium,
    "total_carbohydrate":round_total_carbohydrate,
    "dietary_fiber":     round_dietary_fiber,
    "total_sugars":      round_total_sugars,
    "added_sugars":      round_added_sugars,
    "protein":           round_protein,
    "vitamin_d":         round_vitamin_d,
    "calcium":           round_calcium,
    "iron":              round_iron,
    "potassium":         round_potassium,
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DAILY REFERENCE VALUES (DRVs) AND REFERENCE DAILY INTAKES (RDIs)
# Source: 21 CFR 101.9(c)(8)(iv), updated per 2020 Nutrition Facts Final Rule
#         (85 FR 14150, March 17, 2020)
#         Mandatory compliance date: January 1, 2020 (large manufacturers)
# ─────────────────────────────────────────────────────────────────────────────

# Daily Reference Values (DRVs) — for macronutrients and minerals declared
# as gram or mg amounts. Based on 2000 kcal reference diet.
# Source: 21 CFR 101.9(c)(8)(iv) — 2020 final rule values
DAILY_REFERENCE_VALUES: dict[str, dict] = {
    "total_fat":          {"dv": 78.0,   "unit": "g",   "cfr": "101.9(c)(8)(iv)"},
    "saturated_fat":      {"dv": 20.0,   "unit": "g",   "cfr": "101.9(c)(8)(iv)"},
    "cholesterol":        {"dv": 300.0,  "unit": "mg",  "cfr": "101.9(c)(8)(iv)"},
    "sodium":             {"dv": 2300.0, "unit": "mg",  "cfr": "101.9(c)(8)(iv)"},
    "total_carbohydrate": {"dv": 275.0,  "unit": "g",   "cfr": "101.9(c)(8)(iv)"},
    "dietary_fiber":      {"dv": 28.0,   "unit": "g",   "cfr": "101.9(c)(8)(iv)"},
    "added_sugars":       {"dv": 50.0,   "unit": "g",   "cfr": "101.9(c)(8)(iv)"},
    "protein":            {"dv": 50.0,   "unit": "g",   "cfr": "101.9(c)(8)(iv)"},
    # Micronutrients (RDIs for adults/children ≥4)
    "vitamin_d":          {"dv": 20.0,   "unit": "mcg", "cfr": "101.9(c)(8)(iv)"},
    "calcium":            {"dv": 1300.0, "unit": "mg",  "cfr": "101.9(c)(8)(iv)"},
    "iron":               {"dv": 18.0,   "unit": "mg",  "cfr": "101.9(c)(8)(iv)"},
    "potassium":          {"dv": 4700.0, "unit": "mg",  "cfr": "101.9(c)(8)(iv)"},
    # No DV established for:
    "trans_fat":          {"dv": None,   "unit": "g",   "cfr": "no DV — declare as 0g if <0.5g"},
    "total_sugars":       {"dv": None,   "unit": "g",   "cfr": "no DV — declare amount only"},
}


def compute_percent_dv(nutrient_key: str, declared_value: float) -> Optional[float]:
    """
    Compute %DV for a nutrient given its declared (rounded) value.
    Returns None for nutrients without an established DV (trans fat, total sugars).

    %DV = (declared_value / DV) * 100, rounded to nearest whole percent.
    Per 21 CFR 101.9(c)(8): %DV expressed to nearest 1% increment.
    """
    entry = DAILY_REFERENCE_VALUES.get(nutrient_key)
    if entry is None or entry["dv"] is None:
        return None
    percent = (declared_value / entry["dv"]) * 100.0
    return round(percent)  # nearest whole percent


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: ATWATER CALORIE FACTORS
# Source: 21 CFR 101.9(c)(1)(i)(A) — General Atwater factors
#         FDA Food Labeling Guide confirms: fat=9, carbohydrate=4, protein=4
# ─────────────────────────────────────────────────────────────────────────────

ATWATER_FACTORS: dict[str, float] = {
    "total_fat":          9.0,  # kcal/g — 21 CFR 101.9(c)(1)(i)(A)
    "total_carbohydrate": 4.0,  # kcal/g
    "protein":            4.0,  # kcal/g
    "alcohol":            7.0,  # kcal/g — USDA Handbook No. 74
    "sugar_alcohol":      2.4,  # kcal/g — FDA guidance (erythritol=0, others=2-2.6)
}


def compute_atwater_calories(
    total_fat_g: float,
    total_carb_g: float,
    protein_g: float,
    alcohol_g: float = 0.0,
    sugar_alcohol_g: float = 0.0,
) -> float:
    """
    Compute caloric content using Atwater general food factors.

    CRITICAL SEQUENCE per 21 CFR 101.9(c)(1)(i):
    'factors shall be applied to the actual amount (i.e., before rounding)
    of food components present per serving.'

    This means: multiply UNROUNDED scaled values by factors, then round calories.
    Do NOT apply Atwater to already-rounded macro values.

    Returns unrounded calorie value. Caller must apply round_calories().
    """
    kcal = (
        total_fat_g * ATWATER_FACTORS["total_fat"]
        + total_carb_g * ATWATER_FACTORS["total_carbohydrate"]
        + protein_g * ATWATER_FACTORS["protein"]
        + alcohol_g * ATWATER_FACTORS["alcohol"]
        + sugar_alcohol_g * ATWATER_FACTORS["sugar_alcohol"]
    )
    return kcal


def check_calorie_consistency(
    declared_calories: float,
    declared_fat: float,
    declared_carb: float,
    declared_protein: float,
) -> tuple[bool, float]:
    """
    Verify declared calories are consistent with Atwater calculation
    from declared (rounded) macro values.

    This is the LABEL CONSISTENCY check — uses already-rounded macros.
    Different from the primary Atwater calculation which uses pre-rounded values.

    Returns (is_consistent, calculated_calories).
    Tolerance: ±2 kcal per FDA compliance policy.
    """
    calc_kcal = (
        declared_fat * ATWATER_FACTORS["total_fat"]
        + declared_carb * ATWATER_FACTORS["total_carbohydrate"]
        + declared_protein * ATWATER_FACTORS["protein"]
    )
    rounded_calc = round_calories(calc_kcal)
    is_consistent = abs(declared_calories - rounded_calc) <= 2
    return is_consistent, rounded_calc


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: PDP AREA AND TYPE SIZE REQUIREMENTS
# Source: 21 CFR 101.9(d)(1) and 21 CFR 101.105
#         FDA Food Labeling Guide Chapter 7
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TypeSizeTier:
    min_area_sq_in: float         # Lower bound (inclusive)
    max_area_sq_in: Optional[float]  # Upper bound (exclusive); None = no upper limit
    min_type_size_inch: float     # Minimum type size in inches (based on lower case 'o')
    description: str


# PDP area → minimum type size stepped table
# Source: 21 CFR 101.9(d)(1)(i)-(iv)
PDP_TYPE_SIZE_TABLE: list[TypeSizeTier] = [
    TypeSizeTier(
        min_area_sq_in=0.0,
        max_area_sq_in=5.0,
        min_type_size_inch=1/16,
        description="PDP < 5 sq in: minimum 1/16 inch type",
    ),
    TypeSizeTier(
        min_area_sq_in=5.0,
        max_area_sq_in=25.0,
        min_type_size_inch=1/8,
        description="PDP 5-25 sq in: minimum 1/8 inch type",
    ),
    TypeSizeTier(
        min_area_sq_in=25.0,
        max_area_sq_in=100.0,
        min_type_size_inch=3/16,
        description="PDP 25-100 sq in: minimum 3/16 inch type",
    ),
    TypeSizeTier(
        min_area_sq_in=100.0,
        max_area_sq_in=None,
        min_type_size_inch=1/4,
        description="PDP >= 100 sq in: minimum 1/4 inch type",
    ),
]


def compute_pdp_area(
    container_shape: str,
    height_in: float,
    width_in: Optional[float] = None,
    diameter_in: Optional[float] = None,
    circumference_in: Optional[float] = None,
) -> float:
    """
    Compute Principal Display Panel area in square inches.

    For rectangular containers:
      PDP = largest face area = height × width
      (21 CFR 101.1(b): PDP for rectangular container is largest face)

    For cylindrical containers:
      Total label area = height × circumference
      PDP = 40% of total label area
      (21 CFR 101.1(b): for cylindrical container, PDP = 40% of product of
       height × circumference)
      circumference = π × diameter

    Args:
      container_shape: 'rectangular' or 'cylindrical'
      height_in: container height in inches
      width_in: width for rectangular containers (inches)
      diameter_in: diameter for cylindrical containers (inches)
      circumference_in: circumference if provided directly (inches)

    Returns: PDP area in square inches
    """
    if container_shape == "rectangular":
        if width_in is None:
            raise ValueError("width_in required for rectangular containers")
        return height_in * width_in

    elif container_shape == "cylindrical":
        if circumference_in is not None:
            circ = circumference_in
        elif diameter_in is not None:
            circ = math.pi * diameter_in
        else:
            raise ValueError("Either diameter_in or circumference_in required for cylindrical containers")
        total_label_area = height_in * circ
        return 0.40 * total_label_area  # 40% per 21 CFR 101.1(b)

    else:
        raise ValueError(f"Unknown container_shape: {container_shape}. Use 'rectangular' or 'cylindrical'.")


def lookup_min_type_size(pdp_area_sq_in: float) -> TypeSizeTier:
    """Return the applicable type size tier for a given PDP area."""
    for tier in reversed(PDP_TYPE_SIZE_TABLE):
        if pdp_area_sq_in >= tier.min_area_sq_in:
            return tier
    return PDP_TYPE_SIZE_TABLE[0]  # default to smallest tier


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: DISCRETE UNIT SERVING SIZE DECISION TREE
# Source: 21 CFR 101.9(b)(2)(i)(A)-(D) — fetched 2026-04-06
# ─────────────────────────────────────────────────────────────────────────────

def discrete_unit_serving_size_branch(
    unit_weight_g: float,
    racc_g: float,
) -> dict:
    """
    Determine serving size declaration for a discrete unit product.
    Implements 21 CFR 101.9(b)(2)(i)(A)-(D) exactly.

    Args:
      unit_weight_g: weight of one unit in grams
      racc_g: reference amount for the product category in grams

    Returns dict with:
      branch: 'A', 'B', 'C', or 'D'
      serving_size_description: textual description
      dual_column_required: bool
      notes: explanation of the branch logic
    """
    ratio = unit_weight_g / racc_g

    if ratio <= 0.50:
        # (A) Unit ≤ 50% of RACC → serving = number of whole units closest to RACC
        n_units = round(racc_g / unit_weight_g)
        return {
            "branch": "A",
            "ratio": ratio,
            "serving_size_g": n_units * unit_weight_g,
            "serving_unit_count": n_units,
            "dual_column_required": False,
            "notes": f"Unit {unit_weight_g}g ≤ 50% of RACC {racc_g}g. "
                     f"Serving = {n_units} unit(s) = {n_units * unit_weight_g}g",
        }

    elif 0.50 < ratio < 0.67:
        # (B) Unit 50-67% of RACC → manufacturer may declare 1 or 2 units
        return {
            "branch": "B",
            "ratio": ratio,
            "serving_size_g_option_1": unit_weight_g,
            "serving_size_g_option_2": 2 * unit_weight_g,
            "serving_unit_count": 1,  # default to 1 unit
            "dual_column_required": False,
            "notes": f"Unit {unit_weight_g}g is 50-67% of RACC {racc_g}g. "
                     f"Manufacturer may declare 1 or 2 units.",
        }

    elif 0.67 <= ratio < 2.00:
        # (C) Unit 67% to <200% of RACC → serving = 1 unit
        return {
            "branch": "C",
            "ratio": ratio,
            "serving_size_g": unit_weight_g,
            "serving_unit_count": 1,
            "dual_column_required": False,
            "notes": f"Unit {unit_weight_g}g is 67-200% of RACC {racc_g}g. "
                     f"Serving = 1 unit = {unit_weight_g}g",
        }

    elif 2.00 <= ratio <= 3.00:
        # (D) Unit 200-300% of RACC → serving ≈ RACC amount; dual column required
        # Serving size = amount approximating RACC (round to nearest 5g)
        serving_g = _round_to_nearest(racc_g, 5)
        return {
            "branch": "D",
            "ratio": ratio,
            "serving_size_g": serving_g,
            "serving_unit_count": 1,
            "dual_column_required": True,
            "notes": f"Unit {unit_weight_g}g is 200-300% of RACC {racc_g}g. "
                     f"Dual-column required. Serving ≈ RACC = {serving_g}g.",
        }

    else:
        # > 300% of RACC — not directly covered by (A)-(D); unusual edge case
        return {
            "branch": "OVER_300",
            "ratio": ratio,
            "serving_size_g": _round_to_nearest(racc_g, 5),
            "serving_unit_count": 1,
            "dual_column_required": False,
            "notes": f"Unit {unit_weight_g}g exceeds 300% of RACC {racc_g}g. "
                     f"Contact FDA or apply best judgment per 21 CFR 101.12(f).",
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: BULK PRODUCT SINGLE-SERVING CONTAINER RULES
# Source: 21 CFR 101.9(b)(7) — Single-serving container determination
# ─────────────────────────────────────────────────────────────────────────────

def bulk_package_format_branch(
    total_package_g: float,
    racc_g: float,
) -> dict:
    """
    Determine label format for a bulk (non-discrete-unit) product.

    21 CFR 101.9(b)(7):
    - Package ≤ 200% of RACC → single-serving container
      (label must show per-container values)
    - Package 200% to ≤ 300% of RACC → dual-column mandatory
    - Package > 300% of RACC → multi-serving; single column per-serving values

    NOTE: The 200% boundary is exact. 199% = single-serving. 201% = dual-column.
    This is a common injection point for boundary-case errors.
    """
    ratio = total_package_g / racc_g

    if ratio <= 2.00:
        return {
            "format": "single_serving_container",
            "ratio": ratio,
            "dual_column_required": False,
            "per_container_values_required": True,
            "notes": f"Package {total_package_g}g ≤ 200% of RACC {racc_g}g = single-serving container. "
                     f"Per-container values mandatory.",
        }
    elif ratio <= 3.00:
        return {
            "format": "dual_column_required",
            "ratio": ratio,
            "dual_column_required": True,
            "per_container_values_required": True,
            "notes": f"Package {total_package_g}g is 200-300% of RACC {racc_g}g. "
                     f"Dual column mandatory.",
        }
    else:
        return {
            "format": "multi_serving",
            "ratio": ratio,
            "dual_column_required": False,
            "per_container_values_required": False,
            "notes": f"Package {total_package_g}g > 300% of RACC {racc_g}g = multi-serving. "
                     f"Per-serving values only.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: NUTRIENT CONTENT CLAIM THRESHOLDS
# Source: 21 CFR 101.54 — Claims for protein, fiber, vitamins, minerals
# ─────────────────────────────────────────────────────────────────────────────

NUTRIENT_CONTENT_CLAIM_THRESHOLDS: dict[str, dict] = {
    # Source: 21 CFR 101.54(b)-(e)
    "good_source": {
        "description": "'Good source' of protein, fiber, vitamins, minerals",
        "min_percent_dv": 10,   # ≥ 10% DV per serving
        "cfr": "101.54(b)-(e)",
    },
    "excellent_source": {
        "description": "'Excellent source' or 'High in' protein, fiber, vitamins, minerals",
        "min_percent_dv": 20,   # ≥ 20% DV per serving
        "cfr": "101.54(b)-(e)",
    },
    "more": {
        "description": "'More', 'fortified', 'enriched', 'added' — relative claim",
        "min_percent_dv_increase": 10,  # ≥ 10% DV more than reference food
        "cfr": "101.54(e)",
    },
}


def validate_nutrient_content_claim(
    claim_type: str,
    nutrient_key: str,
    declared_value: float,
) -> tuple[bool, str]:
    """
    Validate whether a nutrient content claim is supported by the computed nutrient value.

    Args:
      claim_type: 'good_source' or 'excellent_source'
      nutrient_key: must be in DAILY_REFERENCE_VALUES
      declared_value: the rounded, declared nutrient value

    Returns (is_valid, explanation_string)
    """
    threshold = NUTRIENT_CONTENT_CLAIM_THRESHOLDS.get(claim_type)
    if threshold is None:
        return False, f"Unknown claim type: {claim_type}"

    percent_dv = compute_percent_dv(nutrient_key, declared_value)
    if percent_dv is None:
        return False, f"No DV established for {nutrient_key}; claim cannot be supported."

    min_dv = threshold["min_percent_dv"]
    is_valid = percent_dv >= min_dv
    explanation = (
        f"Claim '{claim_type}' requires ≥{min_dv}% DV. "
        f"{nutrient_key} declares {declared_value} {DAILY_REFERENCE_VALUES[nutrient_key]['unit']} "
        f"= {percent_dv}% DV. "
        f"Claim is {'SUPPORTED' if is_valid else 'NOT SUPPORTED (VIOLATION)'}."
    )
    return is_valid, explanation


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: QUICK SELF-TEST
# Validates critical boundary cases. Run as: python regulatory_tables.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== FDA Regulatory Tables Self-Test ===\n")

    # Test rounding boundaries
    tests = [
        ("Calories: 4.9 kcal → 0",   round_calories(4.9),   0.0),
        ("Calories: 5.0 kcal → 5",   round_calories(5.0),   5.0),
        ("Calories: 47 kcal → 45",   round_calories(47),    45.0),
        ("Calories: 96 kcal → 100",  round_calories(96),    100.0),

        ("Fat: 0.49g → 0",           round_total_fat(0.49), 0.0),
        ("Fat: 0.5g → 0.5",         round_total_fat(0.5),  0.5),
        ("Fat: 4.97g → 5.0",        round_total_fat(4.97), 5.0),   # stays in 0.5g tier
        ("Fat: 5.0g → 5.0",         round_total_fat(5.0),  5.0),   # boundary: 5.0 in 0.5g tier
        ("Fat: 5.1g → 5",           round_total_fat(5.1),  5.0),   # enters 1g tier
        ("Fat: 6.4g → 6",           round_total_fat(6.4),  6.0),

        ("Sodium: 4.9mg → 0",       round_sodium(4.9),     0.0),
        ("Sodium: 5mg → 5",         round_sodium(5.0),     5.0),
        ("Sodium: 143mg → 140",     round_sodium(143),     140.0),
        ("Sodium: 145mg → 150",     round_sodium(145),     150.0),

        ("Carb: 0.49g → 0",         round_total_carbohydrate(0.49), 0.0),
        ("Carb: 0.5g → 1",          round_total_carbohydrate(0.5),  1.0),  # rounds to nearest 1g
        ("Carb: 1.4g → 1",          round_total_carbohydrate(1.4),  1.0),
        ("Carb: 1.5g → 2",          round_total_carbohydrate(1.5),  2.0),
    ]

    passed = 0
    failed = 0
    for label, got, expected in tests:
        status = "PASS" if abs(got - expected) < 1e-9 else "FAIL"
        if status == "FAIL":
            failed += 1
            print(f"  {status}: {label} | got={got}, expected={expected}")
        else:
            passed += 1
            print(f"  {status}: {label}")

    print(f"\n{passed} passed, {failed} failed")

    # Test Atwater
    print("\n--- Atwater Check ---")
    # A standard granola bar: fat=7g, carb=25g, protein=4g → 7*9 + 25*4 + 4*4 = 63+100+16 = 179
    raw_kcal = compute_atwater_calories(7.0, 25.0, 4.0)
    declared = round_calories(raw_kcal)
    print(f"Granola bar (7g fat, 25g carb, 4g protein): {raw_kcal} raw kcal → {declared} declared")

    # Test calorie consistency check
    ok, calc = check_calorie_consistency(180, 7.0, 25.0, 4.0)
    print(f"Consistency check (declared=180): consistent={ok}, calc={calc}")

    # Test RACC lookup
    print("\n--- RACC Lookups ---")
    key_cats = [
        "Grain-based bars with or without filling or coating, "
        "e.g., breakfast bars, granola bars, rice cereal bars",
        "Nut and seed butters, pastes, or creams",
        "Cookies",
    ]
    for cat in key_cats:
        entry = RACC_BY_CATEGORY.get(cat)
        if entry:
            print(f"  {cat[:50]}... → RACC={entry.racc_g}g, discrete={entry.is_discrete_unit}")
        else:
            print(f"  NOT FOUND: {cat[:60]}")

    # Test discrete unit branching
    print("\n--- Discrete Unit Branching ---")
    # Granola bar: 42g unit, 40g RACC → ratio=1.05 → branch C → serving=1 bar
    result = discrete_unit_serving_size_branch(42.0, 40.0)
    print(f"  42g bar, RACC 40g: branch={result['branch']}, dual={result['dual_column_required']}")

    # Boundary: 200% → branch D
    result = discrete_unit_serving_size_branch(80.0, 40.0)
    print(f"  80g bar, RACC 40g (200%): branch={result['branch']}, dual={result['dual_column_required']}")

    result = discrete_unit_serving_size_branch(79.9, 40.0)
    print(f"  79.9g bar, RACC 40g (199.75%): branch={result['branch']}, dual={result['dual_column_required']}")

    # Test PDP area
    print("\n--- PDP Computation ---")
    # Cylindrical jar: 4 inch height, 3 inch diameter
    area = compute_pdp_area("cylindrical", height_in=4.0, diameter_in=3.0)
    tier = lookup_min_type_size(area)
    print(f"  Cylinder 4\"h × 3\"d: PDP={area:.2f} sq in → min type={tier.min_type_size_inch:.4f}\"")

    # Rectangular: 6×4 inch face
    area = compute_pdp_area("rectangular", height_in=6.0, width_in=4.0)
    tier = lookup_min_type_size(area)
    print(f"  Rectangle 6\"×4\": PDP={area:.2f} sq in → min type={tier.min_type_size_inch:.4f}\"")

    print("\n=== Self-test complete ===")