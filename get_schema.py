from sqlalchemy import create_engine, text, inspect



def getschema_agent(host="localhost", user="root", password="", database="ecommerce"):
    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")
    inspector = inspect(engine)

    lines = []
    for table in inspector.get_table_names():
        lines.append(f"TABLE {table}")
        for col in inspector.get_columns(table):
            name = col["name"]
            ctype = col["type"]
            nullable = "" if col["nullable"] else " NOT_NULL"
            default = f" DEFAULT {col['default']}" if col.get("default") else ""
            lines.append(f"  COLUMN {name} {ctype}{nullable}{default}")

        pk = inspector.get_pk_constraint(table)
        if pk and pk["constrained_columns"]:
            cols = ",".join(pk["constrained_columns"])
            lines.append(f"  PRIMARY_KEY {cols}")

        for fk in inspector.get_foreign_keys(table):
            src = ",".join(fk["constrained_columns"])
            ref = fk["referred_table"]
            col = ",".join(fk["referred_columns"])
            lines.append(f"  FOREIGN_KEY {src} REFERENCES {ref}({col})")

        lines.append("")

    return "\n".join(lines)