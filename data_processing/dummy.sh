python clean.py --language english --input_path data/dummy/dummy.txt --input_format "default" --output_path data/dummy/dummy.vert --output_format "onion" --no_filter
onion -sm -n 5 -p doc -t 0.5 data/dummy/dummy.vert > data/dummy/dummy.onion
onion -sm -n 5 -t 0.8 data/dummy/dummy.vert > data/dummy/dummy.onion
python clean.py --language english --input_path data/dummy/dummy.onion --input_format "onion" --output_path data/dummy/dummy.clean --output_format "default" --no_filter
