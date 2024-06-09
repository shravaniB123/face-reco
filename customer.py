import sqlite3

# Connect to the database
conn = sqlite3.connect('customers.db')
c = conn.cursor()

# Create customers table
c.execute('''CREATE TABLE IF NOT EXISTS customers
             (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, face_encoding BLOB)''')

# Create transactions table
c.execute('''CREATE TABLE IF NOT EXISTS transactions
             (id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id INTEGER, transaction_details TEXT,
              FOREIGN KEY(customer_id) REFERENCES customers(id))''')

conn.commit()
conn.close()

