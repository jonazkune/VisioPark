import 'package:flutter/material.dart';

import '../models/parking.dart';
import '../services/auth_service.dart';
import '../services/detector_service.dart';
import '../services/parking_service.dart';
import 'create_parking_screen.dart';
import 'login_screen.dart';
import 'parking_detail_screen.dart';
import 'users_screen.dart';
import '../widgets/parking_card.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  List<Parking> _parkings = [];
  String? _error;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _reload();
  }

  Future<void> _reload() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final items = await DetectorService.listParkings(AuthService.detectorUrl);
      for (final parking in items) {
        try {
          await ParkingService.upsertParking(parking);
        } catch (_) {}
      }
      if (!mounted) return;
      setState(() {
        _parkings = items;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  void _logout() {
    AuthService.logout();
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(builder: (_) => const LoginScreen()),
      (route) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    final user = AuthService.user;
    return Scaffold(
      appBar: AppBar(
        title: const Text('ParkingAI'),
        actions: [
          IconButton(onPressed: _reload, icon: const Icon(Icons.refresh)),
          IconButton(onPressed: _logout, icon: const Icon(Icons.logout)),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Kaixo, ${user?.name ?? ''} · ${user?.isAdmin == true ? 'aparkaleku guztiak' : 'publikoak eta zure pribatuak'}',
              style: const TextStyle(fontSize: 16, height: 1.4),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                if (user?.canCreateParking == true)
                  ElevatedButton(
                    onPressed: () async {
                      await Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => const CreateParkingScreen(),
                        ),
                      );
                      _reload();
                    },
                    child: const Text('Aparkalekua sortu'),
                  ),
                if (user?.isAdmin == true)
                  ElevatedButton(
                    onPressed: () async {
                      await Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const UsersScreen()),
                      );
                      _reload();
                    },
                    child: const Text('Erabiltzaileak'),
                  ),
              ],
            ),
            const SizedBox(height: 12),
            if (_loading) const LinearProgressIndicator(),
            if (_error != null) Text(_error!),
            Expanded(
              child: _parkings.isEmpty && !_loading
                  ? const Center(child: Text('Ez dago aparkalekurik esleituta.'))
                  : ListView.builder(
                      itemCount: _parkings.length,
                      itemBuilder: (context, index) {
                        final parking = _parkings[index];
                        return ParkingCard(
                          parking: parking,
                          onTap: () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (_) =>
                                    ParkingDetailScreen(parking: parking),
                              ),
                            );
                          },
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
