import 'package:flutter/material.dart';

import '../models/parking.dart';
import 'parking_detail_screen.dart';
import '../services/parking_service.dart';
import '../widgets/parking_card.dart';

class ParkingListScreen extends StatelessWidget {
  const ParkingListScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Aparkalekuen zerrenda')),
      body: StreamBuilder<List<Parking>>(
        stream: ParkingService.getParkings(),
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            return Center(child: Text('Errorea: ${snapshot.error}'));
          }
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          final parkings = snapshot.data ?? [];
          if (parkings.isEmpty) {
            return const Center(child: Text('Ez dago aparkalekurik gordeta.'));
          }
          return ListView.builder(
            itemCount: parkings.length,
            itemBuilder: (context, index) {
              final parking = parkings[index];
              return ParkingCard(
                parking: parking,
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => ParkingDetailScreen(parking: parking),
                    ),
                  );
                },
              );
            },
          );
        },
      ),
    );
  }
}
