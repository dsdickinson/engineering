#!/usr/bin/env python3

import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def fetch_table(debug, url):
	# Fetch the HTML data
	with urllib.request.urlopen(url) as response:
		html = response.read().decode('utf-8')

	# Debug the HTML structure
	if debug == 1:
		print(html)

	# Parse the HTML using BeautifulSoup
	soup = BeautifulSoup(html, 'html.parser')

	# Find table
	table = soup.find_all('table')

	return table

def fetch_data(debug, table):
	# Get table data
	if table:
		table_data = pd.read_html(StringIO(str(table[0])), header=0)[0]

	    # Verify html_table
		if debug == 1:
			print(f'TYPE: {type(table_data)}, SIZE: {len(table_data)}')
			print(table_data)

	else:
		print("\nERROR: No table found! Can not continue.\n")
		exit(1)
  
	return table_data

def create_df(debug, table_data):
	x_coords, characters, y_coords = [], [], []
	x_coords = table_data['x-coordinate']
	characters = table_data['Character']
	y_coords = table_data['y-coordinate']

	if debug == 1:
		print(x_coords)
		print(y_coords)
		print(characters)

	# Create a dataframe so we can sort the data easily
	df = pd.DataFrame({
					'cols': x_coords,  
           	        'chars': characters,
					'rows': y_coords,  
					}) 

	if debug == 1:
		print("Data Frame (not sorted)")
		print(df)

	df_sorted = df.sort_values(by=['rows', 'cols'], ascending=[True, True])
	if debug == 1:
		print("Data Frame (sorted)")
		print(df_sorted)

	return df_sorted

def decode_msg(debug, df_sorted):
	last_row = 0
	track_col = 0
	print("", end="\n")
	for index, row in df_sorted.iterrows():
		this_row = row['rows']
		this_col = row['cols']
		this_char = row['chars']

        # Start a new row
		if this_col == 0 and this_row > last_row:
			print("", end="\n")
			track_col = 0

		# Print empty columns
		while track_col < this_col:
			print(" ", end="")
			track_col += 1

        # Print non-empty columns
		print(this_char, end="")
		track_col = track_col + 1
		last_row = this_row

    # Print new-line at the end
	print("", end="\n")
    
	return True

def main(url):
	debug = 0

	table = fetch_table(debug, url)	
	data = fetch_data(debug, table)	
	df_sorted = create_df(debug, data)
	ret_val = decode_msg(debug, df_sorted)
	print('\nProcess finished with value of {}.\n'.format(ret_val))

if __name__ == "__main__":
    # Source document w/ unicode chars
	url = "https://docs.google.com/document/d/e/2PACX-1vSHesOf9hv2sPOntssYrEdubmMQm8lwjfwv6NPjjmIRYs_FOYXtqrYgjh85jBUebK9swPXh_a5TJ5Kl/pub"

	main(url)
