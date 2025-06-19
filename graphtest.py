from neo4j import GraphDatabase, basic_auth

URI = "bolt://localhost:7687"
username = "neo4j"
password = "password123"

driver = GraphDatabase.driver(URI, auth=basic_auth(username, password))


def create_test_data(tx):
    tx.run("""
        MERGE (a:Person {name: 'Alice'})
        MERGE (b:Person {name: 'Bob'})
        MERGE (a)-[:KNOWS]->(b)
        """)
    
def read_test(tx):
    result = tx.run("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
    for record in result:
        print(f"{record['a.name']} knows {record['b.name']}")

with driver.session() as session:
    session.execute_write(create_test_data)
    session.execute_read(read_test)

driver.close()