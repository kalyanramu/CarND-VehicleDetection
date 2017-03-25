DATA_FILENAME = 'config.json'

data = {'histogram_bins': 32, 'color_space': 'RGB'}
with open(DATA_FILENAME, mode='w', encoding='utf-8') as feedsjson:
    json.dump(data, feedsjson)