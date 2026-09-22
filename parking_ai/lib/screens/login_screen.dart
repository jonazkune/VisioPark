import 'package:flutter/material.dart';

import '../services/detector_service.dart';
import 'home_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _user = TextEditingController();
  final _pass = TextEditingController();
  final _email = TextEditingController();
  final _name = TextEditingController();
  final _pass2 = TextEditingController();
  final _detector = TextEditingController(text: 'http://127.0.0.1:8080');
  bool _busy = false;
  bool _signup = false;
  String? _error;
  String? _info;

  @override
  void dispose() {
    _user.dispose();
    _pass.dispose();
    _email.dispose();
    _name.dispose();
    _pass2.dispose();
    _detector.dispose();
    super.dispose();
  }

  Future<void> _login() async {
    setState(() {
      _busy = true;
      _error = null;
      _info = null;
    });
    try {
      await DetectorService.login(
        baseUrl: _detector.text.trim(),
        username: _user.text.trim(),
        password: _pass.text,
      );
      if (!mounted) return;
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const HomeScreen()),
      );
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _register() async {
    if (_pass.text != _pass2.text) {
      setState(() => _error = 'Pasahitzak ez datoz bat.');
      return;
    }
    setState(() {
      _busy = true;
      _error = null;
      _info = null;
    });
    try {
      final data = await DetectorService.register(
        baseUrl: _detector.text.trim(),
        email: _email.text.trim(),
        password: _pass.text,
        name: _name.text.trim(),
      );
      if (!mounted) return;
      setState(() {
        _signup = false;
        _user.text = _email.text.trim();
        _info = data['preview'] != null
            ? '${data['message'] ?? 'Korreoa ez da bidali.'}\nIreki: ${data['preview']}'
            : (data['message']?.toString() ??
                'Berretsi korreoa saioa hasi aurretik.');
      });
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('ParkingAI · Saioa')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Text(
            _signup
                ? 'Sortu kontua. Korreoa berretsi ondoren parking publikoak ikusiko dituzu.'
                : 'Hasi saioa. Parking publikoak edonork ikusten ditu; pribatuak pribilegioz.',
            style: const TextStyle(fontSize: 16, height: 1.4),
          ),
          const SizedBox(height: 20),
          TextField(
            controller: _detector,
            decoration: const InputDecoration(
              labelText: 'Detektagailua',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          if (_signup) ...[
            TextField(
              controller: _email,
              keyboardType: TextInputType.emailAddress,
              decoration: const InputDecoration(
                labelText: 'Korreoa',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _name,
              decoration: const InputDecoration(
                labelText: 'Izena (aukerakoa)',
                border: OutlineInputBorder(),
              ),
            ),
          ] else
            TextField(
              controller: _user,
              decoration: const InputDecoration(
                labelText: 'Korreoa edo erabiltzailea',
                border: OutlineInputBorder(),
              ),
            ),
          const SizedBox(height: 12),
          TextField(
            controller: _pass,
            obscureText: true,
            decoration: const InputDecoration(
              labelText: 'Pasahitza',
              border: OutlineInputBorder(),
            ),
          ),
          if (_signup) ...[
            const SizedBox(height: 12),
            TextField(
              controller: _pass2,
              obscureText: true,
              decoration: const InputDecoration(
                labelText: 'Errepikatu pasahitza',
                border: OutlineInputBorder(),
              ),
            ),
          ],
          const SizedBox(height: 16),
          ElevatedButton(
            onPressed: _busy ? null : (_signup ? _register : _login),
            child: Text(
              _busy
                  ? (_signup ? 'Bidaltzen...' : 'Sartzen...')
                  : (_signup ? 'Erregistratu' : 'Saioa hasi'),
            ),
          ),
          TextButton(
            onPressed: _busy
                ? null
                : () => setState(() {
                      _signup = !_signup;
                      _error = null;
                      _info = null;
                    }),
            child: Text(_signup ? 'Saioa hasi' : 'Kontua sortu'),
          ),
          if (_info != null) ...[
            const SizedBox(height: 12),
            Text(_info!),
          ],
          if (_error != null) ...[
            const SizedBox(height: 12),
            Text(_error!, style: const TextStyle(color: Colors.red)),
          ],
        ],
      ),
    );
  }
}
