class Sensor:
    def __init__(self, id, name, location, sensor_type, status="active"):
        self.id = id
        self.name = name
        self.location = location
        self.sensor_type = sensor_type
        self.status = status
        
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'location': self.location,
            'sensor_type': self.sensor_type,
            'status': self.status
        } 