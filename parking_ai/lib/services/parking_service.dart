import '../models/parking.dart';
import '../models/space_polygon.dart';
import 'firebase_service.dart';

class ParkingService {
  static final FirebaseService _firebase = FirebaseService();

  static Stream<List<Parking>> getParkings() {
    return _firebase.getCollectionStream('parkings').map((snapshot) {
      return snapshot.docs.map((doc) {
        return Parking.fromFirestore(
          doc.data() as Map<String, dynamic>,
          doc.id,
        );
      }).toList();
    });
  }

  static Stream<Parking> getParkingDetail(String parkingId) {
    return _firebase.getDocumentStream('parkings', parkingId).map((snapshot) {
      final data = snapshot.data();
      if (data == null) {
        throw StateError('Aparkalekua ez da aurkitu');
      }
      return Parking.fromFirestore(data, snapshot.id);
    });
  }

  static Future<void> updateSpaceOccupancy(
    String parkingId,
    String spaceId,
    bool isOccupied,
  ) async {
    await _firebase.updateDocument('parkings', parkingId, {
      'spacesStatus.$spaceId': isOccupied,
    });
  }

  static Future<void> updateOccupancyMap(
    String parkingId,
    Map<String, bool> status,
  ) async {
    if (status.isEmpty) return;
    await _firebase.updateDocument('parkings', parkingId, {
      for (final entry in status.entries)
        'spacesStatus.${entry.key}': entry.value,
    });
  }

  static Future<void> saveSpacePolygons(
    String parkingId,
    Map<String, SpacePolygon> polygons,
  ) async {
    await _firebase.updateDocument('parkings', parkingId, {
      'spacePolygons':
          polygons.map((key, value) => MapEntry(key, value.toMap())),
    });
  }

  static Future<void> updateModelMeta(
    String parkingId, {
    required bool trained,
    required int samples,
  }) async {
    await _firebase.updateDocument('parkings', parkingId, {
      'modelTrained': trained,
      'trainingSamples': samples,
    });
  }

  static Future<String> addParking(Parking parking) async {
    await _firebase.setDocument('parkings', parking.id, parking.toFirestore());
    return parking.id;
  }

  static Future<void> upsertParking(Parking parking) async {
    await _firebase.setDocument('parkings', parking.id, parking.toFirestore());
  }
}
