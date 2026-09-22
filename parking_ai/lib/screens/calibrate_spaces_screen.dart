import 'dart:async';

import 'package:flutter/material.dart';

import '../models/parking.dart';
import '../models/space_polygon.dart';
import '../services/detector_service.dart';
import '../services/parking_service.dart';

class CalibrateSpacesScreen extends StatefulWidget {
  final Parking parking;

  const CalibrateSpacesScreen({super.key, required this.parking});

  @override
  State<CalibrateSpacesScreen> createState() => _CalibrateSpacesScreenState();
}

class _CalibrateSpacesScreenState extends State<CalibrateSpacesScreen> {
  final Map<String, SpacePolygon> _polygons = {};
  final List<SpacePoint> _draft = [];
  Size? _imageSize;
  String _status = 'Kamera irudia kargatzen...';
  bool _busy = false;
  Timer? _refreshTimer;
  final TransformationController _transform = TransformationController();

  double get _zoom {
    final scale = _transform.value.getMaxScaleOnAxis();
    return scale <= 0 ? 1 : scale;
  }

  @override
  void initState() {
    super.initState();
    _polygons.addAll(widget.parking.spacePolygons);
    _prepare();
    _refreshTimer = Timer.periodic(const Duration(seconds: 5), (_) {
      if (mounted) setState(() {});
    });
    _transform.addListener(() {
      if (mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    _transform.dispose();
    super.dispose();
  }

  List<String> get _spaceIds {
    final ids = widget.parking.spacesStatus.keys.toList()..sort();
    return ids;
  }

  String? get _nextId {
    for (final id in _spaceIds) {
      if (!_polygons.containsKey(id)) return id;
    }
    return null;
  }

  Future<void> _prepare() async {
    try {
      await DetectorService.registerParking(
        baseUrl: widget.parking.detectorUrl,
        parkingId: widget.parking.id,
        cameraUrl: widget.parking.cameraUrl,
        spaceIds: _spaceIds,
      );
      if (mounted) {
        setState(() => _status = _hint());
      }
    } catch (e) {
      if (mounted) {
        setState(() => _status = 'Detektagailura ezin konektatu: $e');
      }
    }
  }

  String _hint() {
    final id = _nextId;
    if (id == null) return 'Plaza guztiak markatuta. Gorde.';
    if (_draft.isEmpty) {
      return '$id: klikatu lurreko izkinak. Pinch/scroll = zoom. Irudia 5s-ro berritzen da.';
    }
    if (_draft.length < 3) {
      return '$id: ${_draft.length} puntu. Gutxienez 3, gero Itxi plaza.';
    }
    return '$id: ${_draft.length} puntu. Itxi plaza edo klikatu lehen puntutik gertu.';
  }

  Rect _displayRect(Size widgetSize, Size imageSize) {
    final fitted = applyBoxFit(BoxFit.contain, imageSize, widgetSize);
    final dest = fitted.destination;
    final dx = (widgetSize.width - dest.width) / 2;
    final dy = (widgetSize.height - dest.height) / 2;
    return Rect.fromLTWH(dx, dy, dest.width, dest.height);
  }

  SpacePoint _toNormalized(Offset local, Rect display) {
    return SpacePoint(
      ((local.dx - display.left) / display.width).clamp(0.0, 1.0),
      ((local.dy - display.top) / display.height).clamp(0.0, 1.0),
    );
  }

  Offset _toLocal(SpacePoint point, Rect display) {
    return Offset(
      display.left + point.x * display.width,
      display.top + point.y * display.height,
    );
  }

  void _addPoint(Offset local, Size widgetSize) {
    if (_imageSize == null || _nextId == null) return;
    final display = _displayRect(widgetSize, _imageSize!);
    if (!display.inflate(8).contains(local)) return;
    final point = _toNormalized(local, display);

    if (_draft.length >= 3) {
      final first = _toLocal(_draft.first, display);
      final closePx = 16 / _zoom;
      if ((local - first).distance <= closePx) {
        _closePolygon();
        return;
      }
    }
    setState(() {
      _draft.add(point);
      _status = _hint();
    });
  }

  void _closePolygon() {
    final id = _nextId;
    if (id == null || _draft.length < 3) return;
    setState(() {
      _polygons[id] = SpacePolygon(id: id, points: List.of(_draft));
      _draft.clear();
      _status = _hint();
    });
  }

  void _undo() {
    setState(() {
      if (_draft.isNotEmpty) {
        _draft.removeLast();
      } else if (_polygons.isNotEmpty) {
        _polygons.remove(_polygons.keys.last);
      }
      _status = _hint();
    });
  }

  Future<void> _save() async {
    if (_polygons.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Markatu gutxienez plaza bat.')),
      );
      return;
    }
    setState(() => _busy = true);
    try {
      await ParkingService.saveSpacePolygons(widget.parking.id, _polygons);
      await DetectorService.saveSpaces(
        baseUrl: widget.parking.detectorUrl,
        parkingId: widget.parking.id,
        polygons: _polygons,
      );
      if (mounted) Navigator.pop(context);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Ezin izan da gorde: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final imageUrl = DetectorService.rawUrl(
      widget.parking.detectorUrl,
      widget.parking.id,
    );

    return Scaffold(
      appBar: AppBar(
        title: const Text('Lurreko plazak markatu'),
        actions: [
          IconButton(onPressed: _undo, icon: const Icon(Icons.undo)),
          TextButton(
            onPressed: _draft.length >= 3 ? _closePolygon : null,
            child: const Text('ITXI PLAZA', style: TextStyle(color: Colors.white)),
          ),
          TextButton(
            onPressed: _busy ? null : _save,
            child: const Text('GORDE', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: Text(_status),
          ),
          Expanded(
            child: InteractiveViewer(
              transformationController: _transform,
              minScale: 1,
              maxScale: 8,
              panEnabled: true,
              scaleEnabled: true,
              child: LayoutBuilder(
                builder: (context, constraints) {
                  final widgetSize = Size(constraints.maxWidth, constraints.maxHeight);
                  return GestureDetector(
                    onTapDown: (details) =>
                        _addPoint(details.localPosition, widgetSize),
                    child: SizedBox(
                      width: constraints.maxWidth,
                      height: constraints.maxHeight,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          Image.network(
                            imageUrl,
                            fit: BoxFit.contain,
                            gaplessPlayback: true,
                            errorBuilder: (context, error, stackTrace) => Center(
                              child: Text(
                                'Irudirik ez. Piztu detektagailua eta egiaztatu kamera URL.\n$error',
                                textAlign: TextAlign.center,
                              ),
                            ),
                            frameBuilder: (context, child, frame, wasSync) {
                              if (frame != null && _imageSize == null) {
                                WidgetsBinding.instance.addPostFrameCallback((_) {
                                  _resolveImageSize(imageUrl);
                                });
                              }
                              return child;
                            },
                          ),
                          if (_imageSize != null)
                            CustomPaint(
                              painter: _PolygonPainter(
                                polygons: _polygons,
                                draft: _draft,
                                imageSize: _imageSize!,
                                widgetSize: widgetSize,
                                zoom: _zoom,
                              ),
                            ),
                        ],
                      ),
                    ),
                  );
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _resolveImageSize(String url) async {
    final completer = Completer<Size>();
    final stream = NetworkImage(url).resolve(const ImageConfiguration());
    late final ImageStreamListener listener;
    listener = ImageStreamListener((info, _) {
      completer.complete(
        Size(info.image.width.toDouble(), info.image.height.toDouble()),
      );
      stream.removeListener(listener);
    }, onError: (error, _) {
      if (!completer.isCompleted) completer.completeError(error);
      stream.removeListener(listener);
    });
    stream.addListener(listener);
    try {
      final size = await completer.future;
      if (mounted) setState(() => _imageSize = size);
    } catch (_) {}
  }
}

class _PolygonPainter extends CustomPainter {
  final Map<String, SpacePolygon> polygons;
  final List<SpacePoint> draft;
  final Size imageSize;
  final Size widgetSize;
  final double zoom;

  _PolygonPainter({
    required this.polygons,
    required this.draft,
    required this.imageSize,
    required this.widgetSize,
    this.zoom = 1,
  });

  Rect _display() {
    final fitted = applyBoxFit(BoxFit.contain, imageSize, widgetSize);
    final dest = fitted.destination;
    final dx = (widgetSize.width - dest.width) / 2;
    final dy = (widgetSize.height - dest.height) / 2;
    return Rect.fromLTWH(dx, dy, dest.width, dest.height);
  }

  Offset _map(SpacePoint point, Rect display) {
    return Offset(
      display.left + point.x * display.width,
      display.top + point.y * display.height,
    );
  }

  void _drawPolygon(
    Canvas canvas,
    List<SpacePoint> points,
    Rect display, {
    required bool closed,
    String? label,
  }) {
    final zoom = this.zoom <= 0 ? 1.0 : this.zoom;
    final stroke = (2 / zoom).clamp(0.7, 2.0).toDouble();
    final dot = (5 / zoom).clamp(1.2, 5.0).toDouble();
    final font = (14 / zoom).clamp(8.0, 14.0).toDouble();
    if (points.isEmpty) return;
    final mapped = [for (final point in points) _map(point, display)];
    final path = Path()..moveTo(mapped.first.dx, mapped.first.dy);
    for (final offset in mapped.skip(1)) {
      path.lineTo(offset.dx, offset.dy);
    }
    if (closed) path.close();

    if (closed) {
      canvas.drawPath(
        path,
        Paint()..color = Colors.teal.withValues(alpha: 0.28),
      );
    }
    canvas.drawPath(
      path,
      Paint()
        ..color = Colors.tealAccent
        ..style = PaintingStyle.stroke
        ..strokeWidth = stroke,
    );
    for (var i = 0; i < mapped.length; i++) {
      final offset = mapped[i];
      final isCloseTarget = !closed && i == 0 && points.length >= 3;
      canvas.drawCircle(
        offset,
        isCloseTarget ? dot * 1.35 : dot,
        Paint()..color = isCloseTarget ? Colors.amber.shade200 : Colors.white,
      );
      canvas.drawCircle(
        offset,
        (dot * 0.6).clamp(0.8, dot),
        Paint()..color = Colors.tealAccent,
      );
    }
    if (label != null) {
      final text = TextPainter(
        text: TextSpan(
          text: label,
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
            fontSize: font,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      text.paint(canvas, mapped.first + Offset(6 / zoom, -18 / zoom));
    }
  }

  @override
  void paint(Canvas canvas, Size size) {
    final display = _display();
    for (final polygon in polygons.values) {
      _drawPolygon(
        canvas,
        polygon.points,
        display,
        closed: true,
        label: polygon.id,
      );
    }
    _drawPolygon(canvas, draft, display, closed: false);
  }

  @override
  bool shouldRepaint(covariant _PolygonPainter oldDelegate) => true;
}
