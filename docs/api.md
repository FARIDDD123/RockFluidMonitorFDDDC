# API Documentation

## Base URL

All API endpoints are relative to: `http://your-server-address/api/`

## Endpoints

### Status

```
GET /status
```

Returns the current status of the API.

**Response:**

```json
{
  "status": "online",
  "message": "Rock Fluid Monitor API is running"
}
```

### Sensor Data

```
GET /sensors
```

Returns a list of all sensors.

```
GET /sensors/{id}
```

Returns details for a specific sensor.

```
GET /fluid-data/{sensor_id}
```

Returns fluid data for a specific sensor.

**Query Parameters:**

- `start_time`: ISO datetime string (optional)
- `end_time`: ISO datetime string (optional)
- `limit`: Number of records to return (default: 100)

**Response:**

```json
{
  "data": [
    {
      "sensor_id": "sensor1",
      "timestamp": "2023-07-01T12:00:00Z",
      "pressure": 101.3,
      "temperature": 25.4,
      "viscosity": 1.2,
      "density": 997.0,
      "flow_rate": 2.5
    },
    ...
  ]
}
``` 