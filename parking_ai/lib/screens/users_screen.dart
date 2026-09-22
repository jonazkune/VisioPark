import 'package:flutter/material.dart';

import '../models/app_user.dart';
import '../models/parking.dart';
import '../services/auth_service.dart';
import '../services/detector_service.dart';

class UsersScreen extends StatefulWidget {
  const UsersScreen({super.key});

  @override
  State<UsersScreen> createState() => _UsersScreenState();
}

class _UsersScreenState extends State<UsersScreen> {
  List<AppUser> _users = [];
  List<Parking> _parkings = [];
  final _username = TextEditingController();
  final _name = TextEditingController();
  final _email = TextEditingController();
  String _role = 'user';
  String? _error;

  @override
  void initState() {
    super.initState();
    _reload();
  }

  @override
  void dispose() {
    _username.dispose();
    _name.dispose();
    _email.dispose();
    super.dispose();
  }

  Future<void> _reload() async {
    try {
      final users = await DetectorService.listUsers(AuthService.detectorUrl);
      final parkings = await DetectorService.listParkings(AuthService.detectorUrl);
      if (!mounted) return;
      setState(() {
        _users = users;
        _parkings = parkings;
        _error = null;
      });
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  Future<void> _create() async {
    try {
      await DetectorService.createUser(
        baseUrl: AuthService.detectorUrl,
        username: _username.text.trim(),
        email: _email.text.trim(),
        name: _name.text.trim(),
        role: _role,
      );
      _username.clear();
      _email.clear();
      await _reload();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  Future<void> _assign(String username, String parkingId, bool assigned) async {
    await DetectorService.assignParking(
      baseUrl: AuthService.detectorUrl,
      parkingId: parkingId,
      username: username,
      assigned: assigned,
    );
    await _reload();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Erabiltzaileak')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text('Administratzaileak edo jabeak aparkalekuak esleitzen dizkie erabiltzaileei.'),
          const SizedBox(height: 12),
          TextField(
            controller: _username,
            decoration: const InputDecoration(labelText: 'Erabiltzailea', border: OutlineInputBorder()),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _name,
            decoration: const InputDecoration(labelText: 'Izena', border: OutlineInputBorder()),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _email,
            keyboardType: TextInputType.emailAddress,
            decoration: const InputDecoration(
              labelText: 'Korreoa',
              helperText: 'Aukerakoa: langile kontua. Gidariak beraiek erregistratzen dira.',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 8),
          DropdownButton<String>(
            value: _role,
            items: const [
              DropdownMenuItem(value: 'user', child: Text('Erabiltzailea')),
              DropdownMenuItem(value: 'owner', child: Text('Jabea')),
              DropdownMenuItem(value: 'admin', child: Text('Administratzailea')),
            ],
            onChanged: (value) => setState(() => _role = value ?? 'user'),
          ),
          ElevatedButton(onPressed: _create, child: const Text('Kontua sortu')),
          if (_error != null) Text(_error!, style: const TextStyle(color: Colors.red)),
          const SizedBox(height: 16),
          for (final user in _users)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('${user.name} · ${user.role}', style: const TextStyle(fontWeight: FontWeight.bold)),
                    Text(user.username),
                    for (final parking in _parkings)
                      CheckboxListTile(
                        dense: true,
                        title: Text(parking.name),
                        value: user.parkingIds.contains(parking.id),
                        onChanged: (value) => _assign(user.username, parking.id, value == true),
                      ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
