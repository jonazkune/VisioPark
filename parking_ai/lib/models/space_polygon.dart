class SpacePoint {
  final double x;
  final double y;

  const SpacePoint(this.x, this.y);

  factory SpacePoint.fromMap(Map<String, dynamic> data) {
    return SpacePoint(
      (data['x'] as num?)?.toDouble() ?? 0,
      (data['y'] as num?)?.toDouble() ?? 0,
    );
  }

  Map<String, dynamic> toMap() => {'x': x, 'y': y};
}

class SpacePolygon {
  final String id;
  final List<SpacePoint> points;

  const SpacePolygon({required this.id, required this.points});

  factory SpacePolygon.fromMap(String id, Map<String, dynamic> data) {
    if (data['points'] is List) {
      return SpacePolygon(
        id: id,
        points: [
          for (final point in data['points'] as List)
            SpacePoint.fromMap(Map<String, dynamic>.from(point as Map)),
        ],
      );
    }
    return SpacePolygon(
      id: id,
      points: [
        SpacePoint(
          (data['x1'] as num?)?.toDouble() ?? 0,
          (data['y1'] as num?)?.toDouble() ?? 0,
        ),
        SpacePoint(
          (data['x2'] as num?)?.toDouble() ?? 0,
          (data['y1'] as num?)?.toDouble() ?? 0,
        ),
        SpacePoint(
          (data['x2'] as num?)?.toDouble() ?? 0,
          (data['y2'] as num?)?.toDouble() ?? 0,
        ),
        SpacePoint(
          (data['x1'] as num?)?.toDouble() ?? 0,
          (data['y2'] as num?)?.toDouble() ?? 0,
        ),
      ],
    );
  }

  Map<String, dynamic> toMap() => {
        'points': [for (final point in points) point.toMap()],
      };

  Map<String, dynamic> toDetectorMap() => {
        'id': id,
        'points': [for (final point in points) point.toMap()],
      };
}
