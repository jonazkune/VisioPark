import 'dart:async';

import 'package:flutter/material.dart';

import '../models/parking.dart';
import '../services/auth_service.dart';
import '../services/detector_service.dart';
import '../services/parking_service.dart';
import '../widgets/parking_grid.dart';
import '../widgets/parking_plan.dart';
import 'calibrate_spaces_screen.dart';
import 'train_monitor_screen.dart';

class ParkingDetailScreen extends StatefulWidget {
  final Parking parking;

  const ParkingDetailScreen({super.key, required this.parking});

  @override
  State<ParkingDetailScreen> createState() => _ParkingDetailScreenState();
}

class _ParkingDetailScreenState extends State<ParkingDetailScreen> {
  Timer? _timer;
  ParkingPlan _plan = const ParkingPlan();
  Map<String, bool> _occupancy = {};
  List<String> _occluded = [];
  bool _admin = false;
  bool _showCamera = false;
  String _info = 'Egoera kargatzen...';
  String _cameraUrl = '';

  @override
  void initState() {
    super.initState();
    _occupancy = Map<String, bool>.from(widget.parking.spacesStatus);
    _admin = widget.parking.canEdit || AuthService.user?.isAdmin == true;
    _tick();
    _timer = Timer.periodic(const Duration(seconds: 1), (_) => _tick());
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _tick() async {
    final parking = widget.parking;
    if (parking.detectorUrl.isEmpty) return;
    try {
      final status = await DetectorService.fetchStatus(
        parking.detectorUrl,
        parking.id,
      );
      Map<String, dynamic> planJson = {};
      try {
        planJson = await DetectorService.fetchPlan(
          parking.detectorUrl,
          parking.id,
        );
      } catch (_) {}
      if (status.spaces.isNotEmpty) {
        await ParkingService.updateOccupancyMap(parking.id, status.spaces);
      }
      if (!mounted) return;
      setState(() {
        _occupancy = status.spaces.isNotEmpty ? status.spaces : _occupancy;
        _occluded = status.occluded;
        _plan = ParkingPlan.fromMap(planJson);
        _cameraUrl = DetectorService.rawUrl(parking.detectorUrl, parking.id);
        final free = _occupancy.values.where((v) => !v).length;
        final busy = _occupancy.values.where((v) => v).length;
        _info =
            '$free libre · $busy okupatu'
            '${_occluded.isNotEmpty ? ' · ${_occluded.length} ezkutuan' : ''}'
            '${status.largeVehicles > 0 ? ' · ${status.largeVehicles} ibilgailu handi' : ''}'
            '${status.hiddenFreed.isNotEmpty ? ' · irteera ezkututik' : ''}';
      });
    } catch (e) {
      if (mounted) setState(() => _info = 'Eguneratzeak huts egin du: $e');
    }
  }

  Future<void> _enterAdmin() async {
    if (widget.parking.canEdit || AuthService.user?.isAdmin == true) {
      setState(() => _admin = true);
      return;
    }
    final pin = await showDialog<String>(
      context: context,
      builder: (context) {
        final controller = TextEditingController();
        return AlertDialog(
          title: const Text('Administratzaile PIN'),
          content: TextField(
            controller: controller,
            obscureText: true,
            decoration: const InputDecoration(hintText: 'PIN'),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Utzi'),
            ),
            TextButton(
              onPressed: () => Navigator.pop(context, controller.text),
              child: const Text('Sartu'),
            ),
          ],
        );
      },
    );
    if (pin == null || pin.isEmpty) return;
    try {
      await DetectorService.adminLogin(widget.parking.detectorUrl, pin);
      if (mounted) setState(() => _admin = true);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Sarrera ukatuta: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<Parking>(
      stream: ParkingService.getParkingDetail(widget.parking.id),
      builder: (context, snapshot) {
        final current = snapshot.data ?? widget.parking;
        return Scaffold(
          appBar: AppBar(
            title: Text(current.name),
            actions: [
              TextButton(
                onPressed: _admin ? () => setState(() => _admin = false) : _enterAdmin,
                child: Text(
                  _admin ? 'Irten' : 'Admin',
                  style: const TextStyle(color: Colors.white),
                ),
              ),
            ],
          ),
          body: Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _info,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    if (_admin) ...[
                      const SizedBox(height: 12),
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: [
                          ElevatedButton.icon(
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (_) =>
                                      CalibrateSpacesScreen(parking: current),
                                ),
                              );
                            },
                            icon: const Icon(Icons.crop_free),
                            label: const Text('Kalibratu kamera'),
                          ),
                          ElevatedButton.icon(
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (_) =>
                                      TrainMonitorScreen(parking: current),
                                ),
                              );
                            },
                            icon: const Icon(Icons.model_training),
                            label: const Text('Entrenatu'),
                          ),
                          ElevatedButton.icon(
                            onPressed: () =>
                                setState(() => _showCamera = !_showCamera),
                            icon: const Icon(Icons.videocam),
                            label: Text(
                              _showCamera ? 'Ezkutatu kamera' : 'Ikusi kamera',
                            ),
                          ),
                        ],
                      ),
                    ],
                  ],
                ),
              ),
              if (_admin && _showCamera)
                SizedBox(
                  height: 180,
                  width: double.infinity,
                  child: Image.network(
                    _cameraUrl,
                    fit: BoxFit.contain,
                    gaplessPlayback: true,
                    errorBuilder: (context, error, stackTrace) => const Center(
                      child: Text('Kamera irudirik ez'),
                    ),
                  ),
                ),
              Expanded(
                child: (_plan.isMatrix || _plan.stalls.isNotEmpty)
                    ? ParkingPlanView(
                        plan: _plan,
                        occupancy: _occupancy,
                        occluded: _occluded,
                      )
                    : ParkingGrid(parkingId: current.id),
              ),
            ],
          ),
        );
      },
    );
  }
}
