from pony.orm.core import Entity
from pony import orm
from datetime import datetime
import datetime as dt

db_session = orm.db_session
database_file = "cam.sqlite"
db = orm.Database()
db.bind(provider='sqlite', filename=database_file, create_db=True)


class StaffInfo(db.Entity):
    staff_id = orm.PrimaryKey(int, auto=True)
    name = orm.Required(str)
    embed_vector = orm.Optional(str)
    profile_image = orm.Optional(str)
    records = orm.Set("Record")


class Record(db.Entity):
    staff_id = orm.PrimaryKey(StaffInfo, auto=True)  # cru
    img = orm.Required(str)  # cru
    insert_time = orm.Required(datetime, default=datetime.now)
    capture_time = orm.Required(datetime)  # cru
    flag = orm.Required(bool, default=False)


db.generate_mapping(check_tables=True, create_tables=True)
