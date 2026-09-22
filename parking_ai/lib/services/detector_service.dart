import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/app_user.dart';
import '../models/parking.dart';
import '../models/space_polygon.dart';
import 'auth_service.dart';

class DetectorStatus {
  final bool ok;
  final Map<String, bool> spaces;
  final int vehicles;
  final int largeVehicles;
  final List<String> occluded;
  final List<String> hiddenFreed;
  final bool cameraOk;
  final bool calibrated;
  final bool modelTrained;
  final int samplesFree;
  final int samplesOccupied;
  final String? error;
  final String? updatedAt;

  DetectorStatus({
    required this.ok,
    required this.spaces,
    required this.vehicles,
    this.largeVehicles = 0,
    this.occluded = const [],
    this.hiddenFreed = const [],
    required this.cameraOk,
    required this.calibrated,
    required this.modelTrained,
    required this.samplesFree,
    required this.samplesOccupied,
    this.error,
    this.updatedAt,
  });

  factory DetectorStatus.fromJson(Map<String, dynamic> json) {
    final rawSpaces = json['spaces'] as Map<String, dynamic>? ?? {};
    return DetectorStatus(
      ok: json['ok'] == true,
      spaces: rawSpaces.map((key, value) => MapEntry(key, value == true)),
      vehicles: json['vehicles'] is int ? json['vehicles'] as int : 0,
      largeVehicles: json['large_vehicles'] is int ? json['large_vehicles'] as int : 0,
      occluded: [
        for (final item in (json['occluded'] as List? ?? [])) item.toString(),
      ],
      hiddenFreed: [
        for (final item in (json['hidden_freed'] as List? ?? [])) item.toString(),
      ],
      cameraOk: json['camera_ok'] != false,
      calibrated: json['calibrated'] == true,
      modelTrained: json['model_trained'] == true,
      samplesFree: json['samples_free'] ?? 0,
      samplesOccupied: json['samples_occupied'] ?? 0,
      error: json['error'] as String?,
      updatedAt: json['updated_at'] as String?,
    );
  }
}

class DetectorService {
  static String normalize(String baseUrl) =>
      baseUrl.endsWith('/') ? baseUrl.substring(0, baseUrl.length - 1) : baseUrl;

  static Uri _uri(String baseUrl, String path, [Map<String, String>? query]) {
    final params = <String, String>{
      if (query != null) ...query,
    };
    return Uri.parse('${normalize(baseUrl)}$path').replace(
      queryParameters: params.isEmpty ? null : params,
    );
  }

  static Map<String, String> _headers({bool json = false}) {
    return {
      if (json) 'Content-Type': 'application/json',
      if (AuthService.token != null && AuthService.token!.isNotEmpty)
        'Authorization': 'Bearer ${AuthService.token}',
    };
  }

  static Future<Map<String, dynamic>> _post(
    String baseUrl,
    String path,
    Map<String, dynamic> body,
  ) async {
    final response = await http
        .post(
          _uri(baseUrl, path),
          headers: _headers(json: true),
          body: jsonEncode(body),
        )
        .timeout(const Duration(seconds: 60));
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception(response.body.isEmpty
          ? 'Detektagailuak ${response.statusCode} erantzun du'
          : response.body);
    }
    if (response.body.isEmpty) return {};
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  static Future<Map<String, dynamic>> _get(
    String baseUrl,
    String path, [
    Map<String, String>? query,
  ]) async {
    final response = await http
        .get(_uri(baseUrl, path, query), headers: _headers())
        .timeout(const Duration(seconds: 12));
    if (response.statusCode != 200) {
      throw Exception(response.body.isEmpty
          ? 'Detektagailuak ${response.statusCode} erantzun du'
          : response.body);
    }
    if (response.body.isEmpty) return {};
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  static Future<AppUser> login({
    required String baseUrl,
    required String username,
    required String password,
  }) async {
    final data = await _post(baseUrl, '/auth/login', {
      'username': username,
      'password': password,
    });
    final user = AppUser.fromJson(data['user'] as Map<String, dynamic>);
    AuthService.saveSession(
      detector: baseUrl,
      sessionToken: data['token'] as String,
      sessionUser: user,
    );
    return user;
  }

  static Future<Map<String, dynamic>> register({
    required String baseUrl,
    required String email,
    required String password,
    String name = '',
  }) async {
    return _post(baseUrl, '/auth/register', {
      'email': email,
      'password': password,
      'name': name,
    });
  }

  static Future<List<Parking>> listParkings(String baseUrl) async {
    final data = await _get(baseUrl, '/parkings');
    return [
      for (final item in (data['parkings'] as List? ?? []))
        Parking.fromDetector(Map<String, dynamic>.from(item as Map), baseUrl),
    ];
  }

  static Future<Parking> fetchParking(String baseUrl, String parkingId) async {
    final data = await _get(baseUrl, '/parking', {'parking_id': parkingId});
    return Parking.fromDetector(data, baseUrl);
  }

  static Future<Parking> createParking({
    required String baseUrl,
    required String name,
    String parkingId = '',
    String cameraUrl = '',
    String visibility = 'public',
  }) async {
    final data = await _post(baseUrl, '/parkings', {
      'name': name,
      'parking_id': parkingId,
      'camera_url': cameraUrl,
      'visibility': visibility,
    });
    return Parking.fromDetector(data, baseUrl);
  }

  static Future<List<AppUser>> listUsers(String baseUrl) async {
    final data = await _get(baseUrl, '/users');
    return [
      for (final item in (data['users'] as List? ?? []))
        AppUser.fromJson(Map<String, dynamic>.from(item as Map)),
    ];
  }

  static Future<void> createUser({
    required String baseUrl,
    required String username,
    required String email,
    String name = '',
    String role = 'user',
  }) async {
    await _post(baseUrl, '/users', {
      'username': username,
      'email': email,
      'name': name,
      'role': role,
    });
  }

  static Future<void> assignParking({
    required String baseUrl,
    required String parkingId,
    required String username,
    required bool assigned,
  }) async {
    await _post(baseUrl, '/parkings/assign', {
      'parking_id': parkingId,
      'username': username,
      'assigned': assigned,
    });
  }

  static Future<DetectorStatus> fetchStatus(
    String baseUrl,
    String parkingId,
  ) async {
    final json = await _get(baseUrl, '/status', {'parking_id': parkingId});
    return DetectorStatus.fromJson(json);
  }

  static Future<void> registerParking({
    required String baseUrl,
    required String parkingId,
    required String cameraUrl,
    required List<String> spaceIds,
  }) async {
    await _post(baseUrl, '/register', {
      'parking_id': parkingId,
      'camera_url': cameraUrl,
      'space_ids': spaceIds,
    });
  }

  static Future<void> saveSpaces({
    required String baseUrl,
    required String parkingId,
    required Map<String, SpacePolygon> polygons,
  }) async {
    await _post(baseUrl, '/spaces', {
      'parking_id': parkingId,
      'spaces': [
        for (final polygon in polygons.values) polygon.toDetectorMap(),
      ],
    });
  }

  static Future<Map<String, dynamic>> labelSample({
    required String baseUrl,
    required String parkingId,
    required String spaceId,
    required bool occupied,
  }) {
    return _post(baseUrl, '/label', {
      'parking_id': parkingId,
      'space_id': spaceId,
      'occupied': occupied,
    });
  }

  static Future<Map<String, dynamic>> trainModel({
    required String baseUrl,
    required String parkingId,
  }) {
    return _post(baseUrl, '/train', {'parking_id': parkingId});
  }

  static String snapshotUrl(String baseUrl, String parkingId) {
    return _uri(baseUrl, '/snapshot', {
      'parking_id': parkingId,
      't': DateTime.now().millisecondsSinceEpoch.toString(),
    }).toString();
  }

  static String rawUrl(String baseUrl, String parkingId) {
    return _uri(baseUrl, '/raw', {
      'parking_id': parkingId,
      't': DateTime.now().millisecondsSinceEpoch.toString(),
    }).toString();
  }

  static Future<Map<String, dynamic>> fetchPlan(
    String baseUrl,
    String parkingId,
  ) async {
    return _get(baseUrl, '/plan', {'parking_id': parkingId});
  }

  static Future<void> adminLogin(String baseUrl, String pin) async {
    await _post(baseUrl, '/admin/login', {'pin': pin});
  }
}
