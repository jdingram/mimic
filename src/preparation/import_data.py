'''

This module is for importing the data

'''

import pandas as pd
import psycopg2 as p


def get_table(host, dbname, schema, table, columns, where='row_id>0'):

	'''
	Function for importing the Mimic data from a Postgres database.

	'''

	# Select DB location
	con = p.connect("host={} dbname={}".format(host, dbname))
	cur = con.cursor()

	# Execute query
	cur.execute("SELECT {} FROM {}.{} WHERE {}".format(columns, schema, table, where))
	print('Query executed')

	if cur.rowcount > 0:
		# Get rows and column names
		rows=cur.fetchall()
		column_names = [desc[0] for desc in cur.description]

		# Create DataFrame
		df = pd.DataFrame(rows)
		df.columns = column_names

		# De-Dupe
		df.drop_duplicates(inplace=True)

		print('DF shape is: ' + str(df.shape))
		return df
	else:
		print('No rows returned')
