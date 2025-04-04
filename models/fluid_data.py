from datetime import datetime

class FluidData:
    def __init__(self, sensor_id, timestamp=None, pressure=None, temperature=None, 
                 viscosity=None, density=None, flow_rate=None):
        self.sensor_id = sensor_id
        self.timestamp = timestamp or datetime.now()
        self.pressure = pressure
        self.temperature = temperature
        self.viscosity = viscosity
        self.density = density
        self.flow_rate = flow_rate
        
    def to_dict(self):
        return {
            'sensor_id': self.sensor_id,
            'timestamp': self.timestamp.isoformat(),
            'pressure': self.pressure,
            'temperature': self.temperature,
            'viscosity': self.viscosity,
            'density': self.density,
            'flow_rate': self.flow_rate
        } 