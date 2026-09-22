import 'dart:async';

import 'package:flutter/material.dart';

import '../models/parking.dart';
import '../services/detector_service.dart';
import '../services/parking_service.dart';
import '../widgets/parking_grid.dart';

class TrainMonitorScreen extends StatefulWidget {
  final Parking parking;

  const TrainMonitorScreen({super.key, required this.parking});

  @override
  State<TrainMonitorScreen> createState() => _TrainMonitorScreenState();
}

class _TrainMonitorScreenState extends State<TrainMonitorScreen> {
  Timer? _timer;
  String? _selectedId;
  String _info = 'Detektagailuarekin konektatzen...';
  String _snapshotUrl = '';
  bool _busy = false;
  int _samplesFree = 0;
  int _samplesOccupied = 0;
  bool _trained = false;

  @override
  void initState() {
    super.initState();
    _snapshotUrl = DetectorService.snapshotUrl(
      widget.parking.detectorUrl,
      widget.parking.id,
    );
    _start();
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _start() async {
    try {
      await DetectorService.registerParking(
        baseUrl: widget.parking.detectorUrl,
        parkingId: widget.parking.id,
        cameraUrl: widget.parking.cameraUrl,
        spaceIds: widget.parking.spacesStatus.keys.toList(),
      );
      if (widget.parking.spacePolygons.isNotEmpty) {
        await DetectorService.saveSpaces(
          baseUrl: widget.parking.detectorUrl,
          parkingId: widget.parking.id,
          polygons: widget.parking.spacePolygons,
        );
      }
    } catch (e) {
      if (mounted) setState(() => _info = 'Konektatzeak huts egin du: $e');
    }
    await _tick();
    _timer = Timer.periodic(const Duration(seconds: 4), (_) => _tick());
  }

  Future<void> _tick() async {
    try {
      final status = await DetectorService.fetchStatus(
        widget.parking.detectorUrl,
        widget.parking.id,
      );
      if (status.spaces.isNotEmpty) {
        await ParkingService.updateOccupancyMap(widget.parking.id, status.spaces);
      }
      if (!mounted) return;
      setState(() {
        _samplesFree = status.samplesFree;
        _samplesOccupied = status.samplesOccupied;
        _trained = status.modelTrained;
        _snapshotUrl = DetectorService.snapshotUrl(
          widget.parking.detectorUrl,
          widget.parking.id,
        );
        if (!status.cameraOk) {
          _info = status.error ?? 'Kamera irudirik ez.';
        } else if (!status.calibrated) {
          _info = 'Lehenik kalibratu: marraztu plaza bakoitza kamera honetan.';
        } else if (!status.modelTrained) {
          _info =
              'YOLO orokorra erabiltzen. Entrenatu kamera honetarako: $_samplesFree libre, $_samplesOccupied okupatu.';
        } else {
          _info =
              'Modelo propioa aktibo. Adibideak: $_samplesFree libre, $_samplesOccupied okupatu.';
        }
      });
    } catch (e) {
      if (mounted) setState(() => _info = 'Eguneratzeak huts egin du: $e');
    }
  }

  Future<void> _label(bool occupied) async {
    if (_selectedId == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Aukeratu plaza bat lehenik.')),
      );
      return;
    }
    final id = _selectedId!;
    setState(() {
      _info = '$id → ${occupied ? 'OKUPATUTA' : 'LIBRE'}';
      if (occupied) {
        _samplesOccupied += 1;
      } else {
        _samplesFree += 1;
      }
    });
    ParkingService.updateSpaceOccupancy(widget.parking.id, id, occupied);
    try {
      final result = await DetectorService.labelSample(
        baseUrl: widget.parking.detectorUrl,
        parkingId: widget.parking.id,
        spaceId: id,
        occupied: occupied,
      );
      final total = (result['samples_free'] ?? 0) + (result['samples_occupied'] ?? 0);
      await ParkingService.updateModelMeta(
        widget.parking.id,
        trained: result['model_trained'] == true,
        samples: total is int ? total : 0,
      );
      if (!mounted) return;
      setState(() {
        _samplesFree = result['samples_free'] ?? _samplesFree;
        _samplesOccupied = result['samples_occupied'] ?? _samplesOccupied;
      });
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Adibidea ez da gorde: $e')),
        );
      }
    }
  }

  Future<void> _train() async {
    setState(() => _busy = true);
    try {
      final result = await DetectorService.trainModel(
        baseUrl: widget.parking.detectorUrl,
        parkingId: widget.parking.id,
      );
      await ParkingService.updateModelMeta(
        widget.parking.id,
        trained: true,
        samples: (result['samples'] as int?) ?? (_samplesFree + _samplesOccupied),
      );
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Modeloa entrenatuta. Zehaztasuna: ${result['accuracy'] ?? '-'}',
            ),
          ),
        );
      }
      await _tick();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Entrenamenduak huts egin du: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Entrenatu · ${widget.parking.name}')),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: Text(_info),
          ),
          SizedBox(
            height: 220,
            width: double.infinity,
            child: Image.network(
              _snapshotUrl,
              fit: BoxFit.contain,
              gaplessPlayback: true,
              errorBuilder: (context, error, stackTrace) => Center(
                child: Text(
                  'Irudirik ez. python detector/app.py PCan.\n$error',
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                ElevatedButton(
                  onPressed: _busy ? null : () => _label(false),
                  child: const Text('Errorea: libre da'),
                ),
                ElevatedButton(
                  onPressed: _busy ? null : () => _label(true),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red.shade400,
                    foregroundColor: Colors.white,
                  ),
                  child: const Text('Errorea: okupatuta dago'),
                ),
                ElevatedButton.icon(
                  onPressed: _busy ? null : _train,
                  icon: const Icon(Icons.school),
                  label: Text(_trained ? 'Berriro entrenatu' : 'Entrenatu'),
                ),
              ],
            ),
          ),
          Expanded(
            child: ParkingGrid(
              parkingId: widget.parking.id,
              selectedId: _selectedId,
              onSpaceTap: (id) => setState(() => _selectedId = id),
            ),
          ),
        ],
      ),
    );
  }
}
