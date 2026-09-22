import 'space_polygon.dart';

class Parking {
  final String id;
  final String name;
  final int rows;
  final int cols;
  final Map<String, bool> spacesStatus;
  final Map<String, SpacePolygon> spacePolygons;
  final String cameraUrl;
  final String detectorUrl;
  final bool modelTrained;
  final int trainingSamples;

  final String ownerId;
  final List<String> memberIds;
  final bool canEdit;
  final String visibility;
  final String privilege;

  Parking({
    required this.id,
    required this.name,
    required this.rows,
    required this.cols,
    required this.spacesStatus,
    this.spacePolygons = const {},
    this.cameraUrl = '',
    this.detectorUrl = '',
    this.modelTrained = false,
    this.trainingSamples = 0,
    this.ownerId = '',
    this.memberIds = const [],
    this.canEdit = false,
    this.visibility = 'public',
    this.privilege = 'none',
  });

  int get totalSpaces =>
      spacesStatus.isNotEmpty ? spacesStatus.length : rows * cols;

  int get freeSpaces =>
      spacesStatus.values.where((occupied) => !occupied).length;

  int get occupiedSpaces => totalSpaces - freeSpaces;

  bool get isCalibrated => spacePolygons.isNotEmpty;

  factory Parking.fromFirestore(Map<String, dynamic> data, String id) {
    final rawPolygons = Map<String, dynamic>.from(
      data['spacePolygons'] ?? data['spaceRects'] ?? {},
    );
    return Parking(
      id: id,
      name: data['name'] ?? 'Izenik gabe',
      rows: data['rows'] ?? 1,
      cols: data['cols'] ?? 1,
      spacesStatus: Map<String, bool>.from(data['spacesStatus'] ?? {}),
      spacePolygons: rawPolygons.map(
        (key, value) => MapEntry(
          key,
          SpacePolygon.fromMap(key, Map<String, dynamic>.from(value as Map)),
        ),
      ),
      cameraUrl: data['cameraUrl'] ?? '',
      detectorUrl: data['detectorUrl'] ?? '',
      modelTrained: data['modelTrained'] == true,
      trainingSamples: data['trainingSamples'] ?? 0,
      ownerId: data['ownerId'] ?? '',
      memberIds: [
        for (final item in (data['memberIds'] as List? ?? [])) item.toString(),
      ],
      canEdit: data['canEdit'] == true,
      visibility: data['visibility']?.toString() ?? 'public',
      privilege: data['privilege']?.toString() ?? 'none',
    );
  }

  factory Parking.fromDetector(Map<String, dynamic> data, String detectorUrl) {
    final spacesList = data['spaces'] as List? ?? [];
    final polygons = <String, SpacePolygon>{};
    for (final raw in spacesList) {
      final map = Map<String, dynamic>.from(raw as Map);
      final id = map['id']?.toString() ?? '';
      if (id.isEmpty) continue;
      polygons[id] = SpacePolygon.fromMap(id, map);
    }
    final occupancyRaw = Map<String, dynamic>.from(data['occupancy'] ?? {});
    final status = <String, bool>{
      for (final polygon in polygons.entries)
        polygon.key: occupancyRaw[polygon.key] == true,
    };
    occupancyRaw.forEach((key, value) {
      status[key] = value == true;
    });
    return Parking(
      id: data['id']?.toString() ?? '',
      name: data['name']?.toString() ?? data['id']?.toString() ?? 'Aparkalekua',
      rows: 1,
      cols: polygons.isEmpty ? 1 : polygons.length.clamp(1, 8),
      spacesStatus: status,
      spacePolygons: polygons,
      cameraUrl: data['camera_url']?.toString() ?? '',
      detectorUrl: detectorUrl,
      modelTrained: data['model_trained'] == true,
      trainingSamples: ((data['samples_free'] as num?)?.toInt() ?? 0) +
          ((data['samples_occupied'] as num?)?.toInt() ?? 0),
      ownerId: data['owner_id']?.toString() ?? '',
      memberIds: [
        for (final item in (data['member_ids'] as List? ?? [])) item.toString(),
      ],
      canEdit: data['can_edit'] == true,
      visibility: data['visibility']?.toString() ?? 'public',
      privilege: data['privilege']?.toString() ?? 'none',
    );
  }

  Map<String, dynamic> toFirestore() {
    return {
      'name': name,
      'rows': rows,
      'cols': cols,
      'totalSpaces': totalSpaces,
      'spacesStatus': spacesStatus,
      'spacePolygons':
          spacePolygons.map((key, value) => MapEntry(key, value.toMap())),
      'cameraUrl': cameraUrl,
      'detectorUrl': detectorUrl,
      'modelTrained': modelTrained,
      'trainingSamples': trainingSamples,
      'ownerId': ownerId,
      'memberIds': memberIds,
      'canEdit': canEdit,
      'visibility': visibility,
      'privilege': privilege,
    };
  }
}
