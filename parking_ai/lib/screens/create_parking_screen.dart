import 'package:flutter/material.dart';

import '../services/auth_service.dart';
import '../services/detector_service.dart';
import '../services/parking_service.dart';

class CreateParkingScreen extends StatefulWidget {
  const CreateParkingScreen({super.key});

  @override
  State<CreateParkingScreen> createState() => _CreateParkingScreenState();
}

class _CreateParkingScreenState extends State<CreateParkingScreen> {
  final _nameController = TextEditingController();
  final _idController = TextEditingController();
  final _cameraController = TextEditingController();
  bool _saving = false;
  String _visibility = 'public';

  @override
  void dispose() {
    _nameController.dispose();
    _idController.dispose();
    _cameraController.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    if (_nameController.text.trim().isEmpty) {
      _showMessage('Izena behar da.');
      return;
    }
    setState(() => _saving = true);
    try {
      final parking = await DetectorService.createParking(
        baseUrl: AuthService.detectorUrl,
        name: _nameController.text.trim(),
        parkingId: _idController.text.trim(),
        cameraUrl: _cameraController.text.trim(),
        visibility: _visibility,
      );
      try {
        await ParkingService.upsertParking(parking);
      } catch (_) {}
      if (mounted) Navigator.pop(context);
    } catch (e) {
      _showMessage('Ezin izan da gorde: $e');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  void _showMessage(String text) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Aparkalekua sortu')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text(
            'Publikoa: saioa duen edonork ikusten du okupazioa. Pribatua: pribilegioa behar da.',
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _nameController,
            decoration: const InputDecoration(
              labelText: 'Izena',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _idController,
            decoration: const InputDecoration(
              labelText: 'ID (aukerakoa)',
              hintText: 'proba-ofiziala',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _cameraController,
            decoration: const InputDecoration(
              labelText: 'Kamera IP URL',
              hintText: 'http://192.168.1.64/snapshot.jpg edo rtsp://...',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          DropdownButtonFormField<String>(
            value: _visibility,
            decoration: const InputDecoration(
              labelText: 'Ikusgarritasuna',
              border: OutlineInputBorder(),
            ),
            items: const [
              DropdownMenuItem(
                value: 'public',
                child: Text('Publikoa'),
              ),
              DropdownMenuItem(
                value: 'private',
                child: Text('Pribatua'),
              ),
            ],
            onChanged: (value) =>
                setState(() => _visibility = value ?? 'public'),
          ),
          const SizedBox(height: 24),
          ElevatedButton(
            onPressed: _saving ? null : _save,
            child: Text(_saving ? 'Gordetzen...' : 'Gorde'),
          ),
        ],
      ),
    );
  }
}
