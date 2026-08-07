print("Random Country Generator 🌎")

import pycountry
import random

# List of all country names
countries = [country.name for country in pycountry.countries]
print(random.choice(countries))
